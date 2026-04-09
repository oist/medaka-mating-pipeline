"""
Step 1: load 2D points from processed CSV files.

By default, `use_fish_id=False`, so the points are ordered by score or x
instead of by 2D identity. 3D identity is then decided by geometry in later
steps, which helps prevent 2D ID swaps from leaking into 3D.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

PLAYER_TO_CAM = {"player1": "cam0", "player2": "cam1", "player3": "cam2"}
FISH_ID_ORDER = ["a", "b", "c"]
CAM_VIEW = {"player1": "XY", "player2": "XZ", "player3": "YZ"}


def _normalize_fish_id(fid: Any) -> str:
    """Normalize fish IDs to 'a'/'b'/'c' for comparisons."""
    if fid is None or (isinstance(fid, float) and np.isnan(fid)):
        return ""
    s = str(fid).strip().lower()
    if s in ("a", "b", "c"):
        return s
    if s in ("0", "1", "2"):
        return FISH_ID_ORDER[int(s)]
    return s


def fill_missing_2d_from_two_views(
    cal: Any,
    dfs: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """Fill exactly one missing view from the other two for the same fish/frame."""
    players = ["player1", "player2", "player3"]
    out = {p: dfs[p].copy() for p in players}
    all_frames = set()
    for df in out.values():
        all_frames.update(df["frame"].astype(int).unique())
    frames = sorted(all_frames)

    for frame in frames:
        for pid, df in out.items():
            sub = df[(df["frame"] == frame)]
            if sub.empty:
                continue
            for _, row in sub.iterrows():
                fid = row.get("fish_id")
                key = _normalize_fish_id(fid)
                if key not in ("a", "b", "c"):
                    continue
                pts: Dict[str, Tuple[float, float]] = {}
                for p in players:
                    s = out[p][
                        (out[p]["frame"] == frame)
                        & (out[p]["fish_id"].apply(_normalize_fish_id) == key)
                    ]
                    if not s.empty and pd.notna(s.iloc[0]["x"]) and pd.notna(s.iloc[0]["y"]):
                        pts[p] = (float(s.iloc[0]["x"]), float(s.iloc[0]["y"]))
                    else:
                        pts[p] = None  # type: ignore
                valid = [p for p in players if pts.get(p) is not None]
                if len(valid) != 2:
                    continue
                missing = [p for p in players if p not in valid][0]
                if pid != missing:
                    continue
                try:
                    if missing == "player1":
                        filled = cal.compute_XY_imcoords_from_XZ_YZ(
                            np.array(pts["player2"]), np.array(pts["player3"])
                        )
                    elif missing == "player2":
                        filled = cal.compute_XZ_imcoords_from_XY_YZ(
                            np.array(pts["player1"]), np.array(pts["player3"])
                        )
                    else:
                        filled = cal.compute_YZ_imcoords_from_XZ_XY(
                            np.array(pts["player2"]), np.array(pts["player1"])
                        )
                except Exception:
                    continue
                mask = (
                    (out[missing]["frame"] == frame)
                    & (out[missing]["fish_id"].apply(_normalize_fish_id) == key)
                )
                if mask.any():
                    idx = out[missing].index[mask].tolist()[0]
                    out[missing].loc[idx, "x"] = float(filled[0])
                    out[missing].loc[idx, "y"] = float(filled[1])
    return out


def load_processed_dfs(processed_dir: str, set_key: str) -> Dict[str, pd.DataFrame]:
    """Load processed CSV files for player1/player2/player3."""
    dfs: Dict[str, pd.DataFrame] = {}
    for player in ["player1", "player2", "player3"]:
        path = Path(processed_dir) / f"{set_key}_{player}_processed.csv"
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        df = pd.read_csv(path)
        for col in ["frame", "x", "y"]:
            if col not in df.columns:
                raise ValueError(f"{path} is missing required column: {col}")
        dfs[player] = df
    return dfs


def get_points_per_frame(
    dfs: Dict[str, pd.DataFrame],
    max_points_per_cam: int = 3,
    use_fish_id: bool = False,
) -> Dict[int, Dict[str, List[Tuple[float, float]]]]:
    """Organize 2D points by frame and camera."""
    all_frames = set()
    for df in dfs.values():
        all_frames.update(df["frame"].astype(int).unique())
    frames = sorted(all_frames)

    result: Dict[int, Dict[str, List[Tuple[float, float]]]] = {}
    for frame in frames:
        result[frame] = {}
        for player, df in dfs.items():
            cam = PLAYER_TO_CAM[player]
            sub = df[df["frame"] == frame].dropna(subset=["x", "y"])
            if sub.empty:
                result[frame][cam] = []
                continue
            if use_fish_id and "fish_id" in sub.columns:
                order_map = {fid: i for i, fid in enumerate(FISH_ID_ORDER)}
                sub = sub.copy()
                sub["_ord"] = sub["fish_id"].map(lambda x: order_map.get(x, 999))
                sub = sub.sort_values("_ord").drop(columns=["_ord"])
                sub = sub.head(max_points_per_cam)
            else:
                if "score" in sub.columns:
                    sub = sub.sort_values("score", ascending=False)
                else:
                    sub = sub.sort_values("x", ascending=True)
                sub = sub.head(max_points_per_cam)
            pts = [(float(r["x"]), float(r["y"])) for _, r in sub.iterrows()]
            result[frame][cam] = pts
    return result


def run(
    processed_dir: str,
    set_key: str,
    output_path: str | None = None,
    use_fish_id: bool = False,
) -> Dict:
    """Run Step 1 and optionally write the result to JSON."""
    dfs = load_processed_dfs(processed_dir, set_key)
    points = get_points_per_frame(dfs, use_fish_id=use_fish_id)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in points.items()}, f, indent=2, ensure_ascii=False)
        print(f"Saved: {output_path}")
    return points


def main():
    ap = argparse.ArgumentParser(description="Step 1: load per-frame 2D points from processed CSV files")
    ap.add_argument("--processed_dir", required=True, help="Directory containing processed CSV files")
    ap.add_argument("--set_key", required=True, help="Set key")
    ap.add_argument("--output", default=None, help="Optional output JSON path")
    ap.add_argument(
        "--use_fish_id",
        action="store_true",
        help="Use a/b/c fish_id ordering so 3D IDs stay aligned with 2D IDs",
    )
    args = ap.parse_args()
    run(args.processed_dir, args.set_key, args.output, use_fish_id=args.use_fish_id)


if __name__ == "__main__":
    main()
