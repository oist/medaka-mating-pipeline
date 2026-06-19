"""
Step 5: unify IDs across clips / videos.

The script assigns consistent `ID_1`, `ID_2`, and `ID_3` labels across clips
by matching the last frame of the previous clip to the first frame of the next
clip in 3D space.
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

GROUP_LABELS = ["a", "b", "c"]
ID_LABELS = ["ID_1", "ID_2", "ID_3"]


def load_clip_3d(output_dir: Path, set_key: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load one clip's `a/b/c` 3D CSV files."""
    dfs = {}
    for g in GROUP_LABELS:
        p = output_dir / f"{set_key}_{g}_3d.csv"
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        dfs[g] = pd.read_csv(p)
    return dfs["a"], dfs["b"], dfs["c"]


def get_last_positions(df_a: pd.DataFrame, df_b: pd.DataFrame, df_c: pd.DataFrame):
    """Return the final-frame 3D points ordered as `ID_1/ID_2/ID_3`."""
    out = []
    for df in (df_a, df_b, df_c):
        if len(df) == 0:
            raise ValueError("Clip has no data; cannot read the final frame")
        last = df.iloc[-1]
        out.append([float(last["X"]), float(last["Y"]), float(last["Z"])])
    return np.array(out)


def get_first_positions(df_a: pd.DataFrame, df_b: pd.DataFrame, df_c: pd.DataFrame):
    """Return the first-frame 3D points ordered as `a, b, c`."""
    out = []
    for df in (df_a, df_b, df_c):
        if len(df) == 0:
            raise ValueError("Clip has no data; cannot read the first frame")
        first = df.iloc[0]
        out.append([float(first["X"]), float(first["Y"]), float(first["Z"])])
    return np.array(out)


def match_ids_by_position(prev_pts: np.ndarray, curr_pts: np.ndarray) -> np.ndarray:
    """
    Match current clip labels `a/b/c` to the previous clip's `ID_1/2/3` using
    Hungarian assignment in 3D.
    """
    cost = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            cost[i, j] = np.linalg.norm(prev_pts[i] - curr_pts[j])
    row_ind, col_ind = linear_sum_assignment(cost)
    return np.array(col_ind)


def run(
    output_dir: str,
    set_keys: List[str] | None = None,
) -> List[str]:
    """
    Unify IDs across multiple clips in `output_dir` and write
    `{set_key}_ID_1_3d.csv`, etc.
    """
    out_dir = Path(output_dir)
    if set_keys is None:
        a_files = sorted(out_dir.glob("*_a_3d.csv"))
        set_keys = [f.stem.replace("_a_3d", "") for f in a_files]
    if not set_keys:
        print("No *_a_3d.csv files found; skipping cross-video ID unification.")
        return []

    written = []
    prev_pts = None

    for set_key in set_keys:
        df_a, df_b, df_c = load_clip_3d(out_dir, set_key)
        dfs = [df_a, df_b, df_c]

        if prev_pts is None:
            perm = np.array([0, 1, 2])
            prev_pts = get_last_positions(df_a, df_b, df_c)
        else:
            curr_pts = get_first_positions(df_a, df_b, df_c)
            perm = match_ids_by_position(prev_pts, curr_pts)
            prev_pts = get_last_positions(
                dfs[perm[0]], dfs[perm[1]], dfs[perm[2]]
            )

        for i, id_name in enumerate(ID_LABELS):
            df = dfs[perm[i]].copy()
            path = out_dir / f"{set_key}_{id_name}_3d.csv"
            df.to_csv(path, index=False)
            written.append(str(path))
        print(f"  Wrote {set_key} -> {ID_LABELS}")

    return written


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Step 5: unify ID_1/2/3 across clips")
    ap.add_argument("--output-dir", required=True, help="Directory containing the 3D CSV files")
    ap.add_argument("--set-keys", nargs="*", default=None, help="Set keys in chronological order; if omitted the script scans automatically")
    args = ap.parse_args()
    run(args.output_dir, args.set_keys)


if __name__ == "__main__":
    main()
