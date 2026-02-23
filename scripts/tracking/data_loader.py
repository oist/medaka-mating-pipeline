"""
加载SLEAP输出的CSV文件
"""
import pandas as pd
import numpy as np
from pathlib import Path


def load_sleap_csv(csv_path):
    """
    加载SLEAP CSV文件
    
    Args:
        csv_path: CSV文件路径
        
    Returns:
        DataFrame: 包含frame, track, node, x, y, score列
    """
    df = pd.read_csv(csv_path)
    return df


def parse_sleap_data(df):
    """
    解析SLEAP数据，转换为按帧组织的格式
    
    Args:
        df: SLEAP DataFrame
        
    Returns:
        dict: {
            'frames': array of frame numbers,
            'positions': dict of {frame: {track_id: (x, y, score)}}
        }
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
    加载一个set的三个视角文件
    
    Args:
        file_set_info: dict包含player1_file, player2_file, player3_file
        
    Returns:
        dict: {
            'player1': DataFrame,
            'player2': DataFrame,
            'player3': DataFrame
        }
    """
    data = {}
    for player in ['player1', 'player2', 'player3']:
        file_key = f"{player}_file"
        if file_key in file_set_info:
            csv_path = file_set_info[file_key]
            data[player] = load_sleap_csv(csv_path)
        else:
            raise ValueError(f"缺少 {file_key}")
    
    return data


def get_common_frames(data_dict):
    """
    获取三个视角共同的帧
    
    Args:
        data_dict: {'player1': DataFrame, 'player2': DataFrame, 'player3': DataFrame}
        
    Returns:
        array: 共同的帧号数组
    """
    frames_sets = [set(df['frame'].unique()) for df in data_dict.values()]
    common_frames = set.intersection(*frames_sets)
    return np.array(sorted(list(common_frames)))
