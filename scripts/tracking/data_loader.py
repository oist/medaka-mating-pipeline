"""Load CSV files exported from SLEAP."""
import pandas as pd
import numpy as np
from pathlib import Path


def load_sleap_csv(csv_path):
    """
    Load a SLEAP CSV export.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        DataFrame with frame, track, node, x, y, and score columns.
    """
    df = pd.read_csv(csv_path)
    return df


def _aggregate_track_rows(track_rows):
    """
    Collapse one `(frame, track)` group into a single tracking point.

    SLEAP exports can contain multiple node rows per tracked instance. We use
    the centroid of the finite node coordinates so downstream identity
    assignment sees one point per fish instead of whichever node happened to be
    iterated last.
    """
    xy = track_rows[["x", "y"]].apply(pd.to_numeric, errors="coerce")
    valid_xy = xy.notna().all(axis=1)
    if valid_xy.any():
        x = float(xy.loc[valid_xy, "x"].mean())
        y = float(xy.loc[valid_xy, "y"].mean())
    else:
        x = np.nan
        y = np.nan

    if "score" in track_rows.columns:
        score_series = pd.to_numeric(track_rows["score"], errors="coerce").dropna()
        score = float(score_series.mean()) if not score_series.empty else np.nan
    else:
        score = np.nan

    return x, y, score


def parse_sleap_data(df):
    """
    Convert a SLEAP DataFrame into a frame-indexed structure.

    Args:
        df: SLEAP DataFrame.

    Returns:
        Dictionary with:
        - `frames`: sorted frame numbers
        - `positions`: `{frame: {track_id: (x, y, score)}}`

    When multiple node rows exist for the same `(frame, track)`, they are
    aggregated into one centroid point instead of silently overwriting each
    other.
    """
    if df.empty:
        return {
            'frames': np.array([], dtype=int),
            'positions': {}
        }

    work_df = df.copy()
    work_df['frame'] = pd.to_numeric(work_df['frame'], errors='coerce')
    work_df = work_df.dropna(subset=['frame'])
    work_df['frame'] = work_df['frame'].astype(int)

    frames = np.sort(work_df['frame'].unique())
    
    positions = {}
    for frame in frames:
        frame_data = work_df[work_df['frame'] == frame]
        positions[frame] = {}
        for track_id, track_rows in frame_data.groupby('track', sort=True, dropna=False):
            x, y, score = _aggregate_track_rows(track_rows)
            positions[frame][track_id] = (x, y, score)
    
    return {
        'frames': frames,
        'positions': positions
    }


def load_set_files(file_set_info):
    """
    Load the three view-specific CSV files for one set.

    Args:
        file_set_info: Dictionary containing `player1_file`, `player2_file`,
            and `player3_file`.

    Returns:
        Dictionary with DataFrames for player1/player2/player3.
    """
    data = {}
    for player in ['player1', 'player2', 'player3']:
        file_key = f"{player}_file"
        if file_key in file_set_info:
            csv_path = file_set_info[file_key]
            data[player] = load_sleap_csv(csv_path)
        else:
            raise ValueError(f"Missing required file entry: {file_key}")
    
    return data


def get_common_frames(data_dict):
    """
    Get the frames shared by all three views.

    Args:
        data_dict: `{'player1': DataFrame, 'player2': DataFrame, 'player3': DataFrame}`

    Returns:
        Sorted NumPy array of shared frame numbers.
    """
    frames_sets = [set(df['frame'].unique()) for df in data_dict.values()]
    common_frames = set.intersection(*frames_sets)
    return np.array(sorted(list(common_frames)))
