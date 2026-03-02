"""
Step 1: 点の取得
================

从各相机 processed CSV 中，按帧提取 2D 点（像素坐标）。
默认 use_fish_id=False：不按 2D 身份排序，按 score 或 x 排序，3D 身份由 Step3 几何对应 + Step4 跨帧追踪决定，避免 2D ID 交换。use_fish_id=True 时按 a/b/c 顺序取点，使 3D 与 2D 一致。

可选：fill_missing_2d_from_two_views 用校准模型在「某一帧某个体仅缺 1 个视角」时，
用另两视角预测该视角 2D 并补全；两视角缺失则该帧该个体不补，留待 3D 插值。

用法:
    python -m scripts.reconstruction.step1_get_points --processed_dir H:\\sleap_output\\processed --set_key xxx --output points.json
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

PLAYER_TO_CAM = {"player1": "cam0", "player2": "cam1", "player3": "cam2"}
FISH_ID_ORDER = ["a", "b", "c"]
CAM_VIEW = {"player1": "XY", "player2": "XZ", "player3": "YZ"}  # 用于 2D 补全


def _normalize_fish_id(fid: Any) -> str:
    """将 fish_id 统一为 'a'/'b'/'c' 便于比较。"""
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
    """
    当某一帧某个体仅缺 1 个视角时，用另两视角预测该视角 2D 并写回 DataFrame。
    两视角缺失则该帧该个体不补（留待 3D 阶段插值）；保证 ID 一致。

    cal: 校准对象，需有 compute_XY_imcoords_from_XZ_YZ, compute_XZ_imcoords_from_XY_YZ,
         compute_YZ_imcoords_from_XZ_XY 方法。
    dfs: { "player1": df, "player2": df, "player3": df }，每 df 含 frame, fish_id, x, y。
    返回新的 dfs（不修改入参），补全后的坐标写回对应行列。
    """
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
                    continue  # 只在该缺失视角的当前行上写回补全结果
                try:
                    if missing == "player1":
                        # XY 由 XZ + YZ 预测 → player2, player3
                        filled = cal.compute_XY_imcoords_from_XZ_YZ(
                            np.array(pts["player2"]), np.array(pts["player3"])
                        )
                    elif missing == "player2":
                        # XZ 由 XY + YZ 预测 → player1, player3
                        filled = cal.compute_XZ_imcoords_from_XY_YZ(
                            np.array(pts["player1"]), np.array(pts["player3"])
                        )
                    else:
                        # YZ 由 XZ + XY 预测 → player2, player1
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
    """加载 player1/2/3 的 processed CSV。"""
    dfs: Dict[str, pd.DataFrame] = {}
    for player in ["player1", "player2", "player3"]:
        path = Path(processed_dir) / f"{set_key}_{player}_processed.csv"
        if not path.exists():
            raise FileNotFoundError(f"未找到: {path}")
        df = pd.read_csv(path)
        for col in ["frame", "x", "y"]:
            if col not in df.columns:
                raise ValueError(f"{path} 缺少列: {col}")
        dfs[player] = df
    return dfs


def get_points_per_frame(
    dfs: Dict[str, pd.DataFrame],
    max_points_per_cam: int = 3,
    use_fish_id: bool = False,
) -> Dict[int, Dict[str, List[Tuple[float, float]]]]:
    """
    按帧组织 2D 点。

    use_fish_id=True：有 fish_id 时按 a, b, c 顺序取点（3D 与 2D 身份一致）。
    use_fish_id=False：不使用 2D 身份，只按几何排序（score 或 x），3D 身份由 Step3 几何对应 + Step4 跨帧 3D 追踪决定，避免 2D ID 交换传播到 3D。

    Returns:
        {frame: {cam: [(x,y), ...]}}  每相机最多 max_points_per_cam 个点。
    """
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
            # 仅当 use_fish_id=True 且存在 fish_id 时才按 a/b/c 顺序取点
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
    """
    执行 Step 1，返回点数据。

    use_fish_id=False（推荐）：不按 2D a/b/c 排序，3D 身份由几何+跨帧追踪决定，无 ID 交换。
    use_fish_id=True：按 2D 身份顺序取点，3D 与 2D 个体一致（2D 若有 ID 交换会传到 3D）。
    """
    dfs = load_processed_dfs(processed_dir, set_key)
    points = get_points_per_frame(dfs, use_fish_id=use_fish_id)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in points.items()}, f, indent=2, ensure_ascii=False)
        print(f"已保存: {output_path}")
    return points


def main():
    ap = argparse.ArgumentParser(description="Step 1: 从 processed CSV 获取每帧 2D 点")
    ap.add_argument("--processed_dir", required=True, help="processed CSV 目录")
    ap.add_argument("--set_key", required=True, help="set key")
    ap.add_argument("--output", default=None, help="输出 JSON 路径（可选）")
    ap.add_argument(
        "--use_fish_id",
        action="store_true",
        help="按 a/b/c 顺序取点（3D 与 2D 一致）；默认不启用，3D 身份由几何+跨帧追踪决定",
    )
    args = ap.parse_args()
    run(args.processed_dir, args.set_key, args.output, use_fish_id=args.use_fish_id)


if __name__ == "__main__":
    main()
