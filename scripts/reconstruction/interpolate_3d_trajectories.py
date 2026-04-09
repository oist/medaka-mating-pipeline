"""Interpolate missing frames in 3D trajectories."""

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

GROUP_LABELS = ["a", "b", "c"]


def interpolate_3d_trajectories(
    dfs: Dict[str, pd.DataFrame],
    frame_range: Tuple[int, int] | None = None,
    method: str = "linear",
) -> Dict[str, pd.DataFrame]:
    """Interpolate missing frames in one or more 3D trajectories."""
    if not dfs:
        return {}

    all_frames: set = set()
    for df in dfs.values():
        if df is not None and not df.empty and "frame" in df.columns:
            all_frames.update(df["frame"].astype(int).tolist())
    if frame_range is not None:
        low, high = frame_range
        full_frames = pd.Index(range(low, high + 1), name="frame")
    else:
        if not all_frames:
            return {k: v for k, v in dfs.items() if v is not None and not v.empty}
        full_frames = pd.Index(sorted(all_frames), name="frame")

    result: Dict[str, pd.DataFrame] = {}
    for label in GROUP_LABELS:
        df = dfs.get(label)
        if df is None or df.empty:
            continue
        if "frame" not in df.columns or "X" not in df.columns:
            continue
        df = df[["frame", "X", "Y", "Z"]].copy()
        df = df.drop_duplicates(subset=["frame"]).sort_values("frame")
        df = df.set_index("frame")
        df = df.reindex(full_frames)
        df = df.interpolate(method=method, limit_direction="both")
        df = df.reset_index()
        result[label] = df
    return result


def interpolate_and_save(
    dfs: Dict[str, pd.DataFrame],
    set_key: str,
    output_dir: str,
    *,
    frame_range: Tuple[int, int] | None = None,
    method: str = "linear",
    suffix: str = "_3d_interp",
) -> Dict[str, pd.DataFrame]:
    """Interpolate trajectories and write `{set_key}_a_3d_interp.csv`, etc."""
    interp = interpolate_3d_trajectories(dfs, frame_range=frame_range, method=method)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for label, df in interp.items():
        path = out_dir / f"{set_key}_{label}{suffix}.csv"
        df.to_csv(path, index=False)
        print(f"Saved: {path}")
    return interp
