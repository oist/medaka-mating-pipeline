"""
Simplified 2D tracking pipeline.

This path keeps cross-frame identities stable and ensures three individuals per
frame, but it skips outlier removal and mirror filtering.
"""
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

from .file_set_generator import generate_file_sets, save_file_sets
from .data_loader import load_set_files, parse_sleap_data
from .id_assignment import assign_ids_temporal
from .missing_value_filling import (
    process_all_fish_missing_values,
    extract_time_from_filename,
)
from .visualization import plot_trajectories_three_views


def process_single_set_simple(
    file_set_info,
    sleap_output_dir,
    output_dir=None,
    num_fish=3,
    max_distance=100,
    flow_weight=0.0,
    fps=166,
    interp_method="linear",
    interp_limit=5,
    smooth=True,
    smooth_window=9,
    smooth_order=2,
    visualize=True,
):
    """Run the lightweight pipeline: ID assignment plus missing-value filling."""
    set_key = file_set_info.get("key", "unknown")
    print(f"\nProcessing set (simple pipeline, no outlier removal): {set_key}")

    # 1. Load input data.
    print("Step 1: loading data...")
    data_dict = load_set_files(file_set_info)
    parsed_data = {}
    for player, df in data_dict.items():
        parsed = parse_sleap_data(df)
        parsed["df"] = df
        parsed_data[player] = parsed

    # 2. Assign stable identities across frames.
    print("Step 2: assigning stable IDs across frames...")
    fish_ids = [f"fish_{i}" for i in range(num_fish)]
    assignments = assign_ids_temporal(
        parsed_data,
        num_fish=num_fish,
        max_distance=max_distance,
        flow_weight=flow_weight,
    )

    tracked_dfs = {}
    for player in parsed_data.keys():
        df = parsed_data[player]["df"].copy()
        df["fish_id"] = df.apply(
            lambda row: assignments[player].get(row["frame"], {}).get(
                row["track"], "unassigned"
            ),
            axis=1,
        )
        tracked_dfs[player] = df

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print("Saving ID-assigned CSV files (raw coordinates)...")
        for player, df in tracked_dfs.items():
            path = os.path.join(
                output_dir, f"{set_key}_{player}_simple_id_assigned.csv"
            )
            df.to_csv(path, index=False)
            print(f"  Saved: {path}")

    if visualize:
        print("Rendering visualization after ID assignment...")
        viz_path = (
            os.path.join(output_dir, f"{set_key}_simple_after_id.png")
            if output_dir
            else None
        )
        plot_trajectories_three_views(
            tracked_dfs,
            title_suffix="Simple - After ID Assignment",
            save_path=viz_path,
        )

    # 3. Interpolate missing values.
    print("Step 3: filling missing values (ensure three individuals per frame)...")
    final_dfs = process_all_fish_missing_values(
        tracked_dfs,
        file_set_info=file_set_info,
        fps=fps,
        method=interp_method,
        interp_limit=interp_limit,
        smooth=smooth,
        smooth_window=smooth_window,
        smooth_order=smooth_order,
    )

    if visualize:
        print("Rendering final visualization...")
        final_viz_path = (
            os.path.join(output_dir, f"{set_key}_simple_final.png")
            if output_dir
            else None
        )
        plot_trajectories_three_views(
            final_dfs,
            title_suffix="Simple - Final (no outlier removal)",
            save_path=final_viz_path,
        )

    if output_dir:
        for player, df in final_dfs.items():
            path = os.path.join(output_dir, f"{set_key}_{player}_simple_processed.csv")
            df.to_csv(path, index=False)
            print(f"Saved: {path}")

    return {
        "tracked_dfs": tracked_dfs,
        "final_dfs": final_dfs,
        "assignments": assignments,
    }


def process_from_csv_path_simple(
    csv_path,
    sleap_output_dir,
    output_dir=None,
    num_fish=3,
    max_distance=100,
    flow_weight=0.0,
    fps=166,
    interp_method="linear",
    interp_limit=5,
    smooth=True,
    visualize=True,
    **kwargs,
):
    """Run the simple pipeline starting from a PLAYER1 CSV path."""
    from .file_set_generator import extract_set_key_from_filename, save_file_sets

    set_key = extract_set_key_from_filename(csv_path)
    if not set_key:
        raise ValueError(f"Could not extract a set key from filename: {csv_path}")

    sleap_output_path = Path(sleap_output_dir)
    top_dir = sleap_output_path / "top_mouth_180"
    side_dir = sleap_output_path / "side_mouth_593"
    if not top_dir.exists() or not side_dir.exists():
        inferred = _infer_sleap_output_dir_from_csv(csv_path)
        if inferred:
            sleap_output_dir = inferred
            sleap_output_path = Path(sleap_output_dir)
            top_dir = sleap_output_path / "top_mouth_180"
            side_dir = sleap_output_path / "side_mouth_593"
        if not top_dir.exists() or not side_dir.exists():
            raise ValueError(
                f"Directory does not exist: {top_dir} or {side_dir}. "
                "Please confirm SLEAP_OUTPUT_DIR and your local path configuration."
            )

    file_sets_json_path = os.path.join(sleap_output_dir, "file_sets.json")
    if not os.path.exists(file_sets_json_path):
        file_sets = generate_file_sets(sleap_output_dir)
        save_file_sets(file_sets, file_sets_json_path)
    else:
        with open(file_sets_json_path, "r", encoding="utf-8") as f:
            file_sets = json.load(f)

    if set_key not in file_sets:
        file_sets = generate_file_sets(sleap_output_dir)
        save_file_sets(file_sets, file_sets_json_path)
    if set_key not in file_sets:
        raise ValueError(f"Set key not found: {set_key}")

    file_set_info = file_sets[set_key]
    return process_single_set_simple(
        file_set_info,
        sleap_output_dir,
        output_dir=output_dir,
        num_fish=num_fish,
        max_distance=max_distance,
        flow_weight=flow_weight,
        fps=fps,
        interp_method=interp_method,
        interp_limit=interp_limit,
        smooth=smooth,
        visualize=visualize,
        **kwargs,
    )


def process_from_file_set_key_simple(
    file_set_key,
    file_sets_json_path,
    sleap_output_dir,
    output_dir=None,
    **kwargs,
):
    """Run the simple pipeline for one file set key."""
    with open(file_sets_json_path, "r", encoding="utf-8") as f:
        file_sets = json.load(f)

    if file_set_key not in file_sets:
        raise ValueError(f"Set key not found: {file_set_key}")

    file_set_info = file_sets[file_set_key]
    return process_single_set_simple(
        file_set_info,
        sleap_output_dir,
        output_dir=output_dir,
        **kwargs,
    )


def _infer_sleap_output_dir_from_csv(csv_path):
    """Search upward from a CSV path for a valid SLEAP export root."""
    p = Path(csv_path).resolve()
    for _ in range(10):
        if p.parent == p:
            break
        p = p.parent
        try:
            generate_file_sets(str(p))
            return str(p)
        except (ValueError, OSError):
            continue
    return None
