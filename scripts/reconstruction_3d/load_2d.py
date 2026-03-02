"""
从三视角 *_processed.csv 加载每帧 2D 点，组织为 points_by_frame。
支持单视角少量缺失（该帧该视角用 NaN 表示）。
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# 文件约定：player1=top=cam0, player2=side1=cam1, player3=side2=cam2
PLAYER_TO_CAM = {"player1": "cam0", "player2": "cam1", "player3": "cam2"}
CAM_ORDER = ["cam0", "cam1", "cam2"]
LABELS_ORDER = ["a", "b", "c"]
NUM_INDIVIDUALS = 3


def load_processed_dfs(
    processed_dir: str, set_key: str
) -> Dict[str, pd.DataFrame]:
    """
    加载三视角 processed CSV。
    返回 {player1: df, player2: df, player3: df}。
    """
    root = Path(processed_dir)
    out = {}
    for player in ["player1", "player2", "player3"]:
        path = root / f"{set_key}_{player}_processed.csv"
        if not path.exists():
            raise FileNotFoundError(f"未找到: {path}")
        df = pd.read_csv(path)
        if "x" not in df.columns and "X" in df.columns:
            df = df.rename(columns={"X": "x", "Y": "y"})
        if "x" not in df.columns:
            raise ValueError(f"{path} 需含 x,y 或 X,Y 列")
        out[player] = df
    return out


def _get_one_cam_points_for_frame(
    df: pd.DataFrame,
    frame: int,
    use_fish_id: bool,
) -> List[Tuple[float, float]]:
    """
    从单相机 df 取该帧最多 3 个点，返回 [(x,y), ...]。
    use_fish_id 时按 fish_id a,b,c 顺序；否则按 score 降序、x 升序。
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
    从三视角 DataFrame 组织成每帧、每相机的点列表。
    返回 {frame: {cam0: [(x,y), ...], cam1: [...], cam2: [...]}}
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
    执行加载：读三视角 CSV，返回 points_by_frame。
    """
    dfs = load_processed_dfs(processed_dir, set_key)
    return get_points_per_frame(dfs, use_fish_id=use_fish_id)
