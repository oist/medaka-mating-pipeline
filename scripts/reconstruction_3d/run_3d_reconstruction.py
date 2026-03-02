"""
最精简三视角 3D 重建 + ID 重新分配 + 轨迹可视化
================================================

输入：04_tracking 产出的三个 *_player1/2/3_processed.csv
步骤：加载 2D → 三角化 3D → 每帧选最优对应 → 跨帧 ID（匈牙利）→ 可选缺失帧样条插值 → 导出 CSV → 画轨迹图
输出：{set_key}_3d.csv、{set_key}_3d_trajectories.png
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

from scripts.reconstruction_3d.pipeline import run_pipeline


def run_3d_reconstruction(
    processed_dir: str,
    set_key: str,
    model_folder: str,
    *,
    output_dir: Optional[str] = None,
    xyz_bounds: Tuple[float, float] = (0.0, 200.0),
    fill_missing_3d: bool = True,
    plot: bool = True,
    first_frame_id_only: bool = False,
) -> Dict:
    """
    从三视角 processed CSV 做 3D 重建、跨帧 ID 分配并导出轨迹图。

    Args:
        processed_dir: *_player1/2/3_processed.csv 所在目录
        set_key: 片段 key，与 04_tracking 中一致（如 20251225_K_1_2_clips_...）
        model_folder: 校准模型目录（含 imCoords_to_XYZ.joblib）
        output_dir: 输出目录，默认与 processed_dir 相同
        xyz_bounds: 3D 有效范围 (low, high)
        fill_missing_3d: 是否对缺失帧做三次样条插值，保证每帧都有 3 个个体
        plot: 是否生成 3D 轨迹 PNG
        first_frame_id_only: 若 True，仅用第一帧（或首个有效帧）按 X 定 a→ID_1,b→ID_2,c→ID_3，全片沿用

    Returns:
        {"csv_path": ..., "png_path": ..., "best_tracked": ...}
    """
    out = output_dir or processed_dir
    result = run_pipeline(
        processed_dir=processed_dir,
        set_key=set_key,
        model_folder=model_folder,
        output_dir=out,
        xyz_bounds=xyz_bounds,
        use_fish_id=False,
        shift_to_positive=True,
        save_best_json=False,
        plot=plot,
        fill_missing_3d=fill_missing_3d,
        first_frame_id_only=first_frame_id_only,
    )
    return {
        "csv_path": result["csv_path"],
        "png_path": result["png_path"],
        "best_tracked": result["best_tracked"],
        "points_by_frame": result["points_by_frame"],
    }


def main():
    import argparse
    p = argparse.ArgumentParser(description="三视角 3D 重建 + ID 分配 + 轨迹图")
    p.add_argument("--processed_dir", required=True, help="*_processed.csv 所在目录")
    p.add_argument("--set_key", required=True, help="片段 set key")
    p.add_argument("--model_folder", required=True, help="校准模型目录")
    p.add_argument("--output_dir", default=None, help="输出目录，默认同 processed_dir")
    p.add_argument("--xyz_bounds", type=float, nargs=2, default=[0.0, 200.0], metavar=("LOW", "HIGH"))
    p.add_argument("--no_fill_missing", action="store_true", help="不补全缺失帧")
    p.add_argument("--no_plot", action="store_true", help="不生成轨迹图")
    p.add_argument("--first_frame_id_only", action="store_true", help="仅用第一帧定 ID 映射，全片 a→ID_1,b→ID_2,c→ID_3")
    args = p.parse_args()

    res = run_3d_reconstruction(
        processed_dir=args.processed_dir,
        set_key=args.set_key,
        model_folder=args.model_folder,
        output_dir=args.output_dir,
        xyz_bounds=(args.xyz_bounds[0], args.xyz_bounds[1]),
        fill_missing_3d=not args.no_fill_missing,
        plot=not args.no_plot,
        first_frame_id_only=args.first_frame_id_only,
    )
    if res["csv_path"]:
        print("3D CSV:", res["csv_path"])
    if res["png_path"]:
        print("轨迹图:", res["png_path"])


if __name__ == "__main__":
    main()
