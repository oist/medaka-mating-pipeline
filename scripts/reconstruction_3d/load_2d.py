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


def _normalize_fish_id(fid) -> str:
    """Normalize expected fish IDs to lowercase labels."""
    if fid is None or pd.isna(fid):
        return ""
    return str(fid).strip().lower()


def _aggregate_entity_rows(rows: pd.DataFrame) -> Tuple[float, float, float]:
    """
    Collapse multiple node rows for one tracked entity into a single point.
    """
    xy = rows[["x", "y"]].apply(pd.to_numeric, errors="coerce")
    valid_xy = xy.notna().all(axis=1)
    if valid_xy.any():
        x = float(xy.loc[valid_xy, "x"].mean())
        y = float(xy.loc[valid_xy, "y"].mean())
    else:
        x = np.nan
        y = np.nan

    if "score" in rows.columns:
        score_series = pd.to_numeric(rows["score"], errors="coerce").dropna()
        score = float(score_series.mean()) if not score_series.empty else np.nan
    else:
        score = np.nan
    return x, y, score


def _summarize_entities(
    sub: pd.DataFrame,
    group_col: str | None,
) -> List[Dict[str, object]]:
    """
    Summarize one frame into one point per fish/track instead of per node row.
    """
    if group_col is None:
        entities = []
        for idx, row in sub.iterrows():
            x = float(row["x"])
            y = float(row["y"])
            score = float(row["score"]) if "score" in row and pd.notna(row["score"]) else np.nan
            entities.append({"entity_id": idx, "x": x, "y": y, "score": score})
        return entities

    entities = []
    for entity_id, rows in sub.groupby(group_col, sort=False, dropna=False):
        x, y, score = _aggregate_entity_rows(rows)
        entities.append({"entity_id": entity_id, "x": x, "y": y, "score": score})
    return entities


def _resolve_use_fish_id(
    dfs: Dict[str, pd.DataFrame],
    use_fish_id: bool | None,
) -> bool:
    """Auto-enable fish_id ordering when processed CSVs already contain it."""
    if use_fish_id is not None:
        return bool(use_fish_id)
    return any("fish_id" in df.columns for df in dfs.values())


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
        entities = _summarize_entities(sub, "fish_id")
        by_label: Dict[str, Dict[str, object]] = {}
        extras = []
        for entity in entities:
            label = _normalize_fish_id(entity["entity_id"])
            if label in LABELS_ORDER and label not in by_label:
                by_label[label] = entity
            else:
                extras.append(entity)

        extras.sort(
            key=lambda entity: (
                -(entity["score"] if np.isfinite(entity["score"]) else -np.inf),
                entity["x"] if np.isfinite(entity["x"]) else np.inf,
            )
        )
        points = []
        for fid in LABELS_ORDER:
            entity = by_label.get(fid)
            if entity is None and extras:
                entity = extras.pop(0)
            if entity is None:
                points.append((np.nan, np.nan))
                continue
            x = float(entity["x"])
            y = float(entity["y"])
            points.append((x, y) if np.isfinite(x) and np.isfinite(y) else (np.nan, np.nan))
        return points[:NUM_INDIVIDUALS]

    group_col = "track" if "track" in sub.columns else None
    entities = _summarize_entities(sub, group_col)
    entities.sort(
        key=lambda entity: (
            -(entity["score"] if np.isfinite(entity["score"]) else -np.inf),
            entity["x"] if np.isfinite(entity["x"]) else np.inf,
        )
    )
    points = []
    for entity in entities[:NUM_INDIVIDUALS]:
        x = float(entity["x"])
        y = float(entity["y"])
        points.append((x, y) if np.isfinite(x) and np.isfinite(y) else (np.nan, np.nan))
    while len(points) < NUM_INDIVIDUALS:
        points.append((np.nan, np.nan))
    return points[:NUM_INDIVIDUALS]


def get_points_per_frame(
    dfs: Dict[str, pd.DataFrame],
    use_fish_id: bool | None = None,
) -> Dict[int, Dict[str, List[Tuple[float, float]]]]:
    """
    Convert the three view-specific DataFrames into per-frame point lists.
    """
    use_fish_id = _resolve_use_fish_id(dfs, use_fish_id)
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
    use_fish_id: bool | None = None,
) -> Dict[int, Dict[str, List[Tuple[float, float]]]]:
    """
    Load the processed CSV files and return `points_by_frame`.
    """
    dfs = load_processed_dfs(processed_dir, set_key)
    return get_points_per_frame(dfs, use_fish_id=use_fish_id)
