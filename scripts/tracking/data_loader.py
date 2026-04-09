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


def parse_sleap_data(df):
    """
    Convert a SLEAP DataFrame into a frame-indexed structure.

    Args:
        df: SLEAP DataFrame.

    Returns:
        Dictionary with:
        - `frames`: sorted frame numbers
        - `positions`: `{frame: {track_id: (x, y, score)}}`
    """
    frames = df['frame'].unique()
    frames = np.sort(frames)
    
    positions = {}
    for frame in frames:
        frame_data = df[df['frame'] == frame]
        positions[frame] = {}
        for _, row in frame_data.iterrows():
            track_id = row['track']
            x, y = row['x'], row['y']
            score = row['score']
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
