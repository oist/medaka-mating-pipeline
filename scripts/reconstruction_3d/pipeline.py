"""
三视角 2D → 3D 重建 → 跨帧 ID 追踪 → 导出 3D CSV。
不做缺失填充、不做插值，只输出重建得到的 3D 坐标。
"""

from pathlib import Path
from typing import Dict, List, Tuple

from scripts.reconstruction_3d.calibration import Calibration
from scripts.reconstruction_3d.load_2d import run_load_2d
from scripts.reconstruction_3d.triangulate import compute_candidates
from scripts.reconstruction_3d.select_best import select_best_all_frames
from scripts.reconstruction_3d.cross_frame_tracking import (
    apply_cross_frame_tracking,
    apply_first_frame_id_only,
)
from scripts.reconstruction_3d.fill_missing_3d import (
    best_by_frame_to_dfs,
    dfs_to_single_csv,
    ensure_all_frames_and_fill_spline,
)
from scripts.reconstruction_3d.visualize_3d import run_plot


def run_pipeline(
    processed_dir: str,
    set_key: str,
    model_folder: str,
    *,
    output_dir: str | None = None,
    xyz_bounds: Tuple[float, float] | None = (0.0, 200.0),
    use_fish_id: bool = False,
    shift_to_positive: bool = True,
    save_best_json: bool = False,
    plot: bool = True,
    fill_missing_3d: bool = False,
    first_frame_id_only: bool = False,
) -> Dict:
    """
    跨视角重建 + 跨帧追踪 ID，得到一个 3D CSV。

    1. 加载三视角 2D CSV
    2. 每帧用校准模型重建 3D，选最优几何对应
    3. ID 分配：first_frame_id_only=True 时仅用第一帧（或首个有效帧）按 X 定 a→ID_1,b→ID_2,c→ID_3，全片沿用；
       否则第一帧按 X 赋 ID，后续帧用匈牙利按距离匹配
    4. 导出 CSV：frame, fish_id, X, Y, Z
       - fill_missing_3d=False：仅写有重建结果的帧
       - fill_missing_3d=True：扩展到全部帧，缺失帧用三次样条插值

    output_dir 若给出则写入 {set_key}_3d.csv，可选画轨迹图。
    """
    cal = Calibration(model_folder)
    points_by_frame = run_load_2d(processed_dir, set_key, use_fish_id=use_fish_id)
    candidates_by_frame = compute_candidates(cal, points_by_frame, xyz_bounds=xyz_bounds)
    best = select_best_all_frames(candidates_by_frame, xyz_bounds=xyz_bounds)
    if first_frame_id_only:
        best_tracked = apply_first_frame_id_only(best)
    else:
        best_tracked = apply_cross_frame_tracking(best)

    csv_path = None
    png_path = None
    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if save_best_json:
            import json
            with (out_dir / f"{set_key}_best.json").open("w", encoding="utf-8") as f:
                json.dump({str(k): v for k, v in best_tracked.items()}, f, indent=2, ensure_ascii=False)

        dfs_3d = best_by_frame_to_dfs(
            best_tracked,
            shift_to_positive=shift_to_positive,
            ensure_non_negative=False,  # 空缺用 NaN，不 clip 成 0
        )
        if dfs_3d:
            if fill_missing_3d and points_by_frame:
                all_frames = sorted(points_by_frame.keys())
                dfs_3d = ensure_all_frames_and_fill_spline(dfs_3d, all_frames)
            csv_path = dfs_to_single_csv(dfs_3d, set_key=set_key, output_dir=str(out_dir))
            if plot:
                png_path = run_plot(str(out_dir), set_key, xyz_lim=xyz_bounds)

    return {
        "points_by_frame": points_by_frame,
        "best_tracked": best_tracked,
        "csv_path": csv_path,
        "png_path": png_path,
    }


def discover_sets(processed_dir: str) -> List[str]:
    """扫描目录下 *_player1_processed.csv，返回有 player2/3 配对的 set_key 列表。"""
    root = Path(processed_dir)
    if not root.is_dir():
        return []
    keys = set()
    for p in root.glob("*_player1_processed.csv"):
        name = p.stem.replace("_player1_processed", "")
        if (root / f"{name}_player2_processed.csv").exists() and (root / f"{name}_player3_processed.csv").exists():
            keys.add(name)
    return sorted(keys)
