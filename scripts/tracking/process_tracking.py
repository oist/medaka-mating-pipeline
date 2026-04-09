"""Main 2D post-processing pipeline."""
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

from .file_set_generator import generate_file_sets, save_file_sets
from .data_loader import load_set_files, parse_sleap_data
from .id_assignment import assign_ids_temporal
from .outlier_detection import process_all_fish_outliers, process_mirror_filter
from .missing_value_filling import process_all_fish_missing_values, extract_time_from_filename
from .visualization import plot_trajectories_three_views


def process_single_set(file_set_info, sleap_output_dir, output_dir=None,
                       max_distance=100, max_jump_distance=200, fps=166,
                       interp_method='linear', interp_limit=None, smooth=True,
                       visualize=True, max_iterations=10,
                       use_frame_gap_scaling=True, max_jump_per_frame=100,
                       flow_weight=0, skip_2d_interpolation=False, **kwargs):
    """
    Process one three-view tracking set.
    """
    print(f"\nProcessing set: {file_set_info.get('key', 'unknown')}")
    
    # Step 1: load raw CSV data.
    print("Step 1: loading data...")
    data_dict = load_set_files(file_set_info)
    
    # Parse the loaded data into frame-indexed structures.
    parsed_data = {}
    for player, df in data_dict.items():
        parsed = parse_sleap_data(df)
        parsed['df'] = df
        parsed_data[player] = parsed
    
    # Step 2: assign IDs with Hungarian matching and optional motion cost.
    print("Step 2: assigning IDs...")
    assignments = assign_ids_temporal(
        parsed_data, num_fish=3, max_distance=max_distance, flow_weight=flow_weight
    )
    
    # Build DataFrames with assigned fish IDs.
    tracked_dfs = {}
    for player in parsed_data.keys():
        df = parsed_data[player]['df']
        df = df.copy()
        df['fish_id'] = df.apply(
            lambda row: assignments[player].get(row['frame'], {}).get(row['track'], 'unassigned'),
            axis=1
        )
        tracked_dfs[player] = df
    
    # Save the ID-assigned CSV files before further cleanup.
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        set_key = file_set_info.get('key', 'unknown')
        print("Saving ID-assigned CSV files (raw coordinates)...")
        for player, df in tracked_dfs.items():
            id_assigned_file = os.path.join(output_dir, f"{set_key}_{player}_id_assigned.csv")
            df.to_csv(id_assigned_file, index=False)
            print(f"  Saved: {id_assigned_file}")
    
    # Plot the trajectories after ID assignment.
    if visualize:
        print("Rendering visualization after ID assignment...")
        set_key = file_set_info.get('key', 'unknown')
        id_viz_path = os.path.join(output_dir, f"{set_key}_after_id_assignment.png") if output_dir else None
        plot_trajectories_three_views(tracked_dfs, title_suffix="After ID Assignment", save_path=id_viz_path)
    
    # Step 3: outlier detection using a frame-gap-scaled threshold.
    print("Step 3: removing outliers (single pass with frame-gap scaling)...")
    set_key = file_set_info.get('key', 'unknown')
    cleaned_dfs = process_all_fish_outliers(
        tracked_dfs,
        max_jump_distance=max_jump_distance,
        iterative=True,  # Kept for backward compatibility; ignored internally.
        max_iterations=kwargs.get('max_iterations', max_iterations),
        use_frame_gap_scaling=use_frame_gap_scaling,
        max_jump_per_frame=max_jump_per_frame,
        log_dir=output_dir or sleap_output_dir,
        set_key=set_key,
    )

    # Step 3b: mirror filtering within each frame and view.
    mirror_y_tol = kwargs.get('mirror_y_tol', 20.0)
    mirror_sym_tol = kwargs.get('mirror_sym_tol', 25.0)
    if kwargs.get('apply_mirror_filter', True):
        print("Step 3b: filtering reflection-like mirror pairs...")
        cleaned_dfs = process_mirror_filter(
            cleaned_dfs,
            y_tol=mirror_y_tol,
            sym_tol=mirror_sym_tol,
            log_dir=output_dir or sleap_output_dir,
            set_key=set_key,
        )

    # Step 4: fill missing 2D values unless they should be left for 3D-stage interpolation.
    if skip_2d_interpolation:
        print("Step 4: skipping 2D interpolation (leave gaps for the 3D stage).")
        final_dfs = cleaned_dfs
    else:
        print("Step 4: filling missing values...")
        final_dfs = process_all_fish_missing_values(
            cleaned_dfs,
            file_set_info=file_set_info,
            fps=fps,
            method=interp_method,
            interp_limit=interp_limit,
            smooth=smooth,
            smooth_window=9,
            smooth_order=2
        )
    
    # Plot final processed trajectories.
    if visualize:
        print("Rendering final processed visualization...")
        set_key = file_set_info.get('key', 'unknown')
        final_viz_path = os.path.join(output_dir, f"{set_key}_final.png") if output_dir else None
        plot_trajectories_three_views(final_dfs, title_suffix="Final Processed", save_path=final_viz_path)
    
    # Save final processed CSV files.
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        set_key = file_set_info.get('key', 'unknown')
        
        for player, df in final_dfs.items():
            output_file = os.path.join(output_dir, f"{set_key}_{player}_processed.csv")
            df.to_csv(output_file, index=False)
            print(f"Saved: {output_file}")
    
    return {
        'tracked_dfs': tracked_dfs,
        'final_dfs': final_dfs,
        'assignments': assignments
    }


def process_from_file_set_key(file_set_key, file_sets_json_path, sleap_output_dir, 
                              output_dir=None, **kwargs):
    """
    Process data for one set by key.
    """
    # Load file_sets.json.
    with open(file_sets_json_path, 'r', encoding='utf-8') as f:
        file_sets = json.load(f)
    
    if file_set_key not in file_sets:
        raise ValueError(f"Set key not found: {file_set_key}")
    
    file_set_info = file_sets[file_set_key]
    return process_single_set(file_set_info, sleap_output_dir, output_dir=output_dir, **kwargs)


def _infer_sleap_output_dir_from_csv(csv_path):
    """Search upward from a CSV path for a valid SLEAP export root."""
    from .file_set_generator import generate_file_sets
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


def process_from_csv_path(csv_path, sleap_output_dir, output_dir=None, **kwargs):
    """
    Process data starting from a single CSV path and automatically locate the
    matching three-view set.
    """
    from .file_set_generator import extract_set_key_from_filename, generate_file_sets, save_file_sets

    set_key = extract_set_key_from_filename(csv_path)
    if not set_key:
        raise ValueError(f"Could not extract a set key from filename: {csv_path}")

    sleap_output_path = Path(sleap_output_dir)
    top_dir = sleap_output_path / "top_mouth_180"
    side_dir = sleap_output_path / "side_mouth_593"
    if not top_dir.exists() or not side_dir.exists():
        inferred = _infer_sleap_output_dir_from_csv(csv_path)
        if inferred:
            print(f"Inferred SLEAP export root from CSV path: {inferred}")
            sleap_output_dir = inferred
        else:
            raise ValueError(
                f"Directory does not exist: {top_dir} or {side_dir}. "
                "Please confirm that SLEAP_OUTPUT_DIR points to the actual "
                "export root (for example D:\\sleap_output)."
            )

    file_sets_json_path = os.path.join(sleap_output_dir, "file_sets.json")
    if not os.path.exists(file_sets_json_path):
        print("Generating file_sets.json...")
        file_sets = generate_file_sets(sleap_output_dir)
        save_file_sets(file_sets, file_sets_json_path)
    else:
        with open(file_sets_json_path, 'r', encoding='utf-8') as f:
            file_sets = json.load(f)

    if set_key not in file_sets:
        print(f"Set key not found: {set_key}. Regenerating file_sets.json...")
        file_sets = generate_file_sets(sleap_output_dir)
        save_file_sets(file_sets, file_sets_json_path)

    if set_key not in file_sets:
        raise ValueError(f"Set key not found: {set_key}. Check the filename format.")

    file_set_info = file_sets[set_key]
    return process_single_set(file_set_info, sleap_output_dir, output_dir=output_dir, **kwargs)
