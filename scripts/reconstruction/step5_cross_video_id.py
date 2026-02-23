"""
Step 5: 跨视频/跨片段统一 ID（ID_1, ID_2, ID_3）
================================================

在多个片段（clip）的 a/b/c 3D 轨迹上，为同一物理个体赋予跨片段一致的 ID_1, ID_2, ID_3。
假定片段顺序为时间顺序；用上一片段最后一帧与当前片段第一帧的 3D 位置做 Hungarian 匹配，
再按匹配结果重写当前片段的 CSV 为 _ID_1_3d.csv, _ID_2_3d.csv, _ID_3_3d.csv。

输入: 同一 output_dir 下多组 {set_key}_a_3d.csv, _b_3d.csv, _c_3d.csv
输出: 同目录下 {set_key}_ID_1_3d.csv, _ID_2_3d.csv, _ID_3_3d.csv（首片段用 a→ID_1, b→ID_2, c→ID_3）
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

GROUP_LABELS = ["a", "b", "c"]
ID_LABELS = ["ID_1", "ID_2", "ID_3"]


def load_clip_3d(output_dir: Path, set_key: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """加载一个片段的 a/b/c 3D CSV，返回 (df_a, df_b, df_c)。"""
    dfs = {}
    for g in GROUP_LABELS:
        p = output_dir / f"{set_key}_{g}_3d.csv"
        if not p.exists():
            raise FileNotFoundError(f"未找到: {p}")
        dfs[g] = pd.read_csv(p)
    return dfs["a"], dfs["b"], dfs["c"]


def get_last_positions(df_a: pd.DataFrame, df_b: pd.DataFrame, df_c: pd.DataFrame):
    """返回最后一帧的 3 个 3D 点，顺序为 [ID_1 对应, ID_2 对应, ID_3 对应]。"""
    out = []
    for df in (df_a, df_b, df_c):
        if len(df) == 0:
            raise ValueError("片段无数据，无法取最后一帧")
        last = df.iloc[-1]
        out.append([float(last["X"]), float(last["Y"]), float(last["Z"])])
    return np.array(out)


def get_first_positions(df_a: pd.DataFrame, df_b: pd.DataFrame, df_c: pd.DataFrame):
    """返回第一帧的 3 个 3D 点，顺序 a, b, c。"""
    out = []
    for df in (df_a, df_b, df_c):
        if len(df) == 0:
            raise ValueError("片段无数据，无法取第一帧")
        first = df.iloc[0]
        out.append([float(first["X"]), float(first["Y"]), float(first["Z"])])
    return np.array(out)


def match_ids_by_position(prev_pts: np.ndarray, curr_pts: np.ndarray) -> np.ndarray:
    """
    prev_pts: (3, 3) 上一片段最后一帧 ID_1, ID_2, ID_3 的 XYZ
    curr_pts: (3, 3) 当前片段第一帧 a, b, c 的 XYZ
    返回: (3,) 的 permutation，curr_perm[i] = 当前片段中对应 ID_i 的 label 索引 (0=a,1=b,2=c)
    """
    cost = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            cost[i, j] = np.linalg.norm(prev_pts[i] - curr_pts[j])
    row_ind, col_ind = linear_sum_assignment(cost)
    # row_ind: ID 索引 0,1,2  col_ind: 当前 a,b,c 索引
    # ID_i 对应 current 的 col_ind[i]
    return np.array(col_ind)


def run(
    output_dir: str,
    set_keys: List[str] | None = None,
) -> List[str]:
    """
    对 output_dir 下多组 a/b/c 3D CSV 做跨片段 ID 统一，写出 _ID_1_3d.csv 等。
    set_keys: 片段 key 列表（时间顺序）；若为 None 则自动扫描 *_a_3d.csv 并按 key 排序。
    """
    out_dir = Path(output_dir)
    if set_keys is None:
        a_files = sorted(out_dir.glob("*_a_3d.csv"))
        set_keys = [f.stem.replace("_a_3d", "") for f in a_files]
    if not set_keys:
        print("未找到任何 *_a_3d.csv，跳过跨视频 ID。")
        return []

    written = []
    # 上一片段最后一帧的 3 个点（按 ID_1, ID_2, ID_3）
    prev_pts = None

    for set_key in set_keys:
        df_a, df_b, df_c = load_clip_3d(out_dir, set_key)
        dfs = [df_a, df_b, df_c]

        if prev_pts is None:
            # 第一个片段：固定 a→ID_1, b→ID_2, c→ID_3
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
        print(f"  已写 {set_key} -> {ID_LABELS}")

    return written


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Step 5: 跨视频统一 ID_1/2/3")
    ap.add_argument("--output-dir", required=True, help="3D CSV 所在目录")
    ap.add_argument("--set-keys", nargs="*", default=None, help="按时间顺序的 set_key 列表，不填则自动扫描")
    args = ap.parse_args()
    run(args.output_dir, args.set_keys)


if __name__ == "__main__":
    main()
