"""
Load per-frame 2D points from the three `*_processed.csv` files and organize
them into `points_by_frame`.

Small per-view gaps are allowed and represented as `NaN`.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# File convention: player1=top=cam0, player2=side1=cam1, player3=side2=cam2
PLAYER_TO_CAM = {"player1": "cam0", "player2": "cam1", "player3": "cam2"}
CAM_ORDER = ["cam0", "cam1", "cam2"]
LABELS_ORDER = ["a", "b", "c"]
NUM_INDIVIDUALS = 3


def load_processed_dfs(
    processed_dir: str, set_key: str
) -> Dict[str, pd.DataFrame]:
    """
    Load the three processed CSV files for one clip.
    """
    root = Path(processed_dir)
    out = {}
    for player in ["player1", "player2", "player3"]:
        path = root / f"{set_key}_{player}_processed.csv"
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        df = pd.read_csv(path)
        if "x" not in df.columns and "X" in df.columns:
            df = df.rename(columns={"X": "x", "Y": "y"})
        if "x" not in df.columns:
            raise ValueError(f"{path} must contain x,y or X,Y columns")
        out[player] = df
    return out


def _get_one_cam_points_for_frame(
    df: pd.DataFrame,
    frame: int,
    use_fish_id: bool,
) -> List[Tuple[float, float]]:
    """
    Get up to three 2D points for one frame from one camera DataFrame.

    If `use_fish_id=True`, points are returned in `a,b,c` fish-id order.
    Otherwise they are sorted by score descending, then x ascending.
    """
    sub = df[df["frame"] == frame].copy()
    if sub.empty:
        return []

    if "fish_id" in sub.columns and use_fish_id:
        order_ids = [c for c in LABELS_ORDER if c in sub["fish_id"].values]
        order_ids += [x for x in sub["fish_id"].unique() if x not in order_ids]
        points = []
        for fid in order_ids[:NUM_INDIVIDUALS]:
            row = sub[sub["fish_id"] == fid]
            if not row.empty:
                r = row.iloc[0]
                x, y = float(r["x"]), float(r["y"])
                points.append((x, y) if np.isfinite(x) and np.isfinite(y) else (np.nan, np.nan))
            else:
                points.append((np.nan, np.nan))
        return points[:NUM_INDIVIDUALS]

    if "score" in sub.columns:
        sub = sub.sort_values(by=["score"], ascending=[False])
    else:
        sub = sub.sort_values(by=["x"])
    points = []
    for _, r in sub.head(NUM_INDIVIDUALS).iterrows():
        x, y = float(r["x"]), float(r["y"])
        points.append((x, y) if np.isfinite(x) and np.isfinite(y) else (np.nan, np.nan))
    while len(points) < NUM_INDIVIDUALS:
        points.append((np.nan, np.nan))
    return points[:NUM_INDIVIDUALS]


def get_points_per_frame(
    dfs: Dict[str, pd.DataFrame],
    use_fish_id: bool = False,
) -> Dict[int, Dict[str, List[Tuple[float, float]]]]:
    """
    Convert the three view-specific DataFrames into per-frame point lists.
    """
    all_frames = set()
    for df in dfs.values():
        if "frame" in df.columns:
            all_frames.update(df["frame"].astype(int).tolist())
    frames = sorted(all_frames) if all_frames else []

    result: Dict[int, Dict[str, List[Tuple[float, float]]]] = {}
    for f in frames:
        result[f] = {}
        for player, cam in PLAYER_TO_CAM.items():
            df = dfs.get(player)
            if df is None:
                result[f][cam] = [(np.nan, np.nan)] * NUM_INDIVIDUALS
            else:
                result[f][cam] = _get_one_cam_points_for_frame(df, f, use_fish_id)
                while len(result[f][cam]) < NUM_INDIVIDUALS:
                    result[f][cam].append((np.nan, np.nan))
                result[f][cam] = result[f][cam][:NUM_INDIVIDUALS]
    return result


def run_load_2d(
    processed_dir: str,
    set_key: str,
    use_fish_id: bool = False,
) -> Dict[int, Dict[str, List[Tuple[float, float]]]]:
    """
    Load the processed CSV files and return `points_by_frame`.
    """
    dfs = load_processed_dfs(processed_dir, set_key)
    return get_points_per_frame(dfs, use_fish_id=use_fish_id)
