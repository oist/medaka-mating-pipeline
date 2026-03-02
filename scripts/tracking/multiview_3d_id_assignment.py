"""
多视角 3D 基于几何的一致 ID 分配（ステップ1→2→3 一体版）
==========================================================

目的
----
在三个相机视角同时看到 3 条鱼的场景中，利用事先训练好的 2D→3D 校准模型，
按以下 3 个ステップ，为每一帧选择 **几何上最自然** 的 a/b/c 3 个 3D 个体：

1. 点の取得（Step 1）
   - 从每个相机的 `*_processed.csv` 中提取每帧的 2D 像素坐标。
2. 3次元候補の計算（Step 2）
   - 使用校准模型，对所有 cam0×cam1×cam2 的 2D 组合计算 3D 候选点与配准误差。
3. 最適な対応の選択（Step 3）
   - 在所有 3D 候选中，选择 3 个互不冲突的组合，使总配准误差最小，对应 a/b/c 三条鱼。

本模块是对 `scripts.reconstruction.step1_get_points/step2_compute_3d_candidates/step3_select_best_correspondence`
的轻量封装，提供一个「一键跑完三步」的 API，方便在 tracking / notebook 里调用。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from scripts.reconstruction.step1_get_points import (
    load_processed_dfs,
    get_points_per_frame,
    fill_missing_2d_from_two_views,
)
from scripts.reconstruction.step2_compute_3d_candidates import (
    compute_candidates,
)
from scripts.reconstruction.step3_select_best_correspondence import (
    select_best_all_frames,
)
from scripts.reconstruction.step4_cross_frame_tracking import apply_cross_frame_tracking
from scripts.reconstruction.export_3d_groups import best_by_frame_to_dfs, dfs_to_single_csv
from scripts.reconstruction.fill_missing_3d import ensure_all_frames_and_fill_spline
from scripts.reconstruction.interpolate_3d_trajectories import interpolate_and_save
from scripts.reconstruction.visualize_3d import run_plot

# 视角名与 Step1 的 cam 对应（用于 best_2d 与视频路径）
VIEW_TO_CAM = {"top": "cam0", "side1": "cam1", "side2": "cam2"}
LABELS = ["a", "b", "c"]


def _build_best_2d_per_frame(
    points_by_frame: Dict[int, Dict[str, List[Tuple[float, float]]]],
    best: Dict[int, Dict],
) -> Dict[int, Dict[str, Dict[str, List[float]]]]:
    """
    从 Step1 的 points_by_frame 与 Step3 的 best（含 perm_cam1/perm_cam2）
    得到每帧、每视角、每条鱼对应的 2D 像素坐标（即用于计算该帧 3D 的 2D 点）。

    Returns:
        {frame: {"top": {"a": [x,y], "b": [...], "c": [...]}, "side1": {...}, "side2": {...}}}
    """
    out: Dict[int, Dict[str, Dict[str, List[float]]]] = {}
    for frame, info in best.items():
        groups = info.get("groups")
        perm_cam1 = info.get("perm_cam1")
        perm_cam2 = info.get("perm_cam2")
        if not groups or perm_cam1 is None or perm_cam2 is None:
            continue
        pts = points_by_frame.get(frame)
        if not pts:
            continue
        cam0 = pts.get("cam0", [])
        cam1 = pts.get("cam1", [])
        cam2 = pts.get("cam2", [])
        k = len(groups)
        frame_2d: Dict[str, Dict[str, List[float]]] = {}
        for view, cam in VIEW_TO_CAM.items():
            if cam == "cam0":
                idx_list = list(range(k))
            elif cam == "cam1":
                idx_list = perm_cam1
            else:
                idx_list = perm_cam2
            arr = pts[cam]
            frame_2d[view] = {}
            for i, label in enumerate(LABELS):
                if i >= len(idx_list):
                    continue
                idx = idx_list[i]
                if idx >= len(arr):
                    continue
                if label not in groups:
                    continue
                x, y = arr[idx]
                frame_2d[view][label] = [float(x), float(y)]
        out[frame] = frame_2d
    return out


def run_multiview_id_pipeline(
    processed_dir: str,
    set_key: str,
    model_folder: str,
    *,
    output_dir: str | None = None,
    xyz_bounds: Tuple[float, float] | None = (0.0, 200.0),
    use_fish_id: bool = False,
    save_best_json: bool = True,
    shift_to_positive: bool = False,
    interpolate_3d: bool = True,
    fill_missing_2d: bool = True,
    cross_frame_tracking: bool = True,
    flow_weight: float = 10.0,
    fill_missing_3d: bool = True,
) -> Dict:
    """
    一次性跑完「ステップ1～4」，基于 3D 几何为每帧选择最自然的 a/b/c 对应，并做跨帧 ID 一致化。

    典型使用场景：
        - 已经用 `scripts.tracking.process_tracking` 得到了三视角的 `*_processed.csv`
        - 想利用 3D 几何重新做一次「干净的 a/b/c 分配」，避免单视角 ID 交换

    Step 3 每帧独立选最小配准误差的 a/b/c，会导致帧间 ID 乱跳；Step 4 用匈牙利算法 + 运动方向
    对 Step 3 结果做跨帧关联，保证同一物理个体在时间上保持同一标签。导出 CSV 与 best_2d 使用 Step 4 结果。

    Args:
        processed_dir:
            `*_processed.csv` 所在目录，需包含
                {set_key}_player1_processed.csv
                {set_key}_player2_processed.csv
                {set_key}_player3_processed.csv
        set_key:
            当前片段的 key（与 processed/tracking 中保持一致）
        model_folder:
            校准模型目录，需与 `scripts.reconstruction.calibration.Calibration`
            的约定一致（含 imCoords_to_XYZ.joblib 等）
        output_dir:
            若非 None，则把 3D 结果导出到该目录：
                - {set_key}_best.json   （每帧的 a/b/c 3D 结果）
                - {set_key}_3d.csv     （单 CSV，列 frame, fish_id, X, Y, Z，与 2D 一致）
        xyz_bounds:
            3D 有效范围 (LOW, HIGH)，X/Y/Z 任一超出则候选视为无效；
            例如 (0, 200) 或 (0, 1200)。若为 None 则不做范围裁剪。
        use_fish_id:
            传给 Step 1：True 时按 2D `fish_id` a/b/c 顺序取点；
            False（推荐）时只按 score/x 排序，由 3D 几何决定最终 a/b/c。
        save_best_json:
            若 True 且 output_dir 非 None，则保存 {set_key}_best.json。
        shift_to_positive:
            传给导出函数：若为 True，将所有 3D 坐标整体平移到非负区间。
        interpolate_3d:
            若 True 且 output_dir 非 None，在导出 3D CSV 后对 a/b/c 轨迹做缺失帧线性插值，
            并写出 {set_key}_a_3d_interp.csv 等。
        fill_missing_2d:
            若 True，在 Step1 前用校准模型做「2D 补全」：某一帧某个体仅缺 1 个视角时，
            用另两视角预测该视角 2D 并补全；两视角缺失则不补，留待 Step6 3D 插值。
        cross_frame_tracking:
            若 True（推荐），在 Step 3 后执行 Step 4 跨帧追踪，使同一物理个体在时间上保持同一 a/b/c 标签；
            否则仅用 Step 3 的每帧最优（易出现每帧 ID 不同）。
        flow_weight:
            跨帧追踪时运动方向惩罚权重（mm），用于匈牙利匹配代价；方向突变时增加代价，默认 10.0。
        fill_missing_3d:
            若 True（默认），在导出前将每条轨迹扩展到全部帧并用**三次样条**补全缺失帧，
            保证最终 CSV 每帧都有 3 个个体坐标（不用线性插值，曲线更自然）。

    Returns:
        字典，包含：
            - "points_by_frame": Step1 的输出 {frame: {cam: [(x,y),...]}}
            - "candidates_by_frame": Step2 的输出
            - "best": Step3 的输出 {frame: {"groups": {"a": [X,Y,Z], ...}}}
            - "csv_path": 若导出了 3D CSV，则为单文件路径；否则为 None
    """
    processed_dir_path = Path(processed_dir)
    if not processed_dir_path.exists():
        raise FileNotFoundError(f"processed_dir 不存在: {processed_dir_path}")

    from scripts.reconstruction.calibration import Calibration

    cal = Calibration(model_folder)

    # ---- Step 1: 点の取得（从 processed CSV 组织每帧每相机的 2D 点）----
    dfs = load_processed_dfs(str(processed_dir_path), set_key)
    if fill_missing_2d and "fish_id" in next(iter(dfs.values())).columns:
        dfs = fill_missing_2d_from_two_views(cal, dfs)
    points_by_frame = get_points_per_frame(dfs, use_fish_id=use_fish_id)

    # ---- Step 2: 3次元候補の計算（枚举 cam0×cam1×cam2 组合并计算 3D 候选与 cost）----
    candidates_by_frame = compute_candidates(cal, points_by_frame, xyz_bounds=xyz_bounds)

    # ---- Step 3: 最適な対応の選択（在所有候选中选出 a/b/c，每帧独立）----
    best = select_best_all_frames(candidates_by_frame, xyz_bounds=xyz_bounds)

    # ---- Step 4: 跨帧个体追踪（匈牙利 + flow），保证同一物理个体时间上标签一致 ----
    if cross_frame_tracking:
        best_for_export = apply_cross_frame_tracking(best, flow_weight=flow_weight)
    else:
        best_for_export = best

    # ---- 与 3D 对应的每视角 2D 点（用于 2D 视频叠加；使用跨帧追踪后的标签）----
    best_2d_per_frame = _build_best_2d_per_frame(points_by_frame, best_for_export)

    csv_paths: Dict[str, str] = {}
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 保存 best.json（可选）：保存跨帧追踪后的结果，供 CSV/视频使用
        if save_best_json:
            import json

            best_path = out_dir / f"{set_key}_best.json"
            with best_path.open("w", encoding="utf-8") as f:
                json.dump({str(k): v for k, v in best_for_export.items()}, f, indent=2, ensure_ascii=False)

        # 从跨帧追踪后的 best 构建 per-label 的 DataFrame（不写三文件）
        dfs_3d = best_by_frame_to_dfs(best_for_export, shift_to_positive=shift_to_positive)
        all_frames = sorted(points_by_frame.keys())
        if fill_missing_3d and dfs_3d and all_frames:
            dfs_3d = ensure_all_frames_and_fill_spline(dfs_3d, all_frames)
        # 导出为单 CSV：frame, fish_id, X, Y, Z（与 2D 一致）
        if dfs_3d:
            single_path = dfs_to_single_csv(dfs_3d, set_key=set_key, output_dir=str(out_dir))
            csv_paths["3d"] = single_path

        if interpolate_3d and dfs_3d:
            from scripts.reconstruction.interpolate_3d_trajectories import interpolate_and_save
            interpolate_and_save(
                dfs_3d,
                set_key=set_key,
                output_dir=str(out_dir),
                suffix="_3d_interp",
            )

        # 保存与 3D 对应的 2D 点（供 2×2 视频叠加用）
        if best_2d_per_frame:
            import json
            best_2d_path = out_dir / f"{set_key}_best_2d.json"
            with best_2d_path.open("w", encoding="utf-8") as f:
                json.dump({str(k): v for k, v in best_2d_per_frame.items()}, f, indent=2, ensure_ascii=False)

    return {
        "points_by_frame": points_by_frame,
        "candidates_by_frame": candidates_by_frame,
        "best": best,
        "best_tracked": best_for_export,
        "best_2d_per_frame": best_2d_per_frame,
        "csv_paths": csv_paths,
    }


def quick_plot_3d_trajectories(
    output_dir: str,
    set_key: str,
    xyz_bounds: Tuple[float, float] | None = (0.0, 200.0),
) -> str:
    """
    辅助函数：从本模块导出的 3D CSV 生成一张简单的 3D 轨迹 PNG。

    相当于对 `scripts.reconstruction.visualize_3d.run_plot` 的轻量封装。

    Args:
        output_dir: `run_multiview_id_pipeline` 导出 3D CSV 的目录
        set_key:    片段 key
        xyz_bounds: 3D 视图的坐标范围 (low, high)，与校准保持一致，默认 (0, 200)

    Returns:
        生成的 PNG 路径字符串。
    """
    png_path = run_plot(output_dir, set_key, xyz_lim=xyz_bounds)
    return png_path


def main():
    """命令行入口：一键跑完三视角 3D ID 分配。"""
    import argparse

    ap = argparse.ArgumentParser(
        description=(
            "多视角 3D 基于几何的一致 ID 分配："
            "从 processed CSV 读取 2D 点，跑 Step1/2/3，导出 a/b/c 三条 3D 轨迹。"
        )
    )
    ap.add_argument("--processed_dir", required=True, help="*_processed.csv 所在目录")
    ap.add_argument("--set_key", required=True, help="片段 set key，与 tracking 中一致")
    ap.add_argument("--model_folder", required=True, help="校准模型目录（含 imCoords_to_XYZ.joblib 等）")
    ap.add_argument("--output_dir", required=True, help="3D 结果输出目录")
    ap.add_argument(
        "--xyz_bounds",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=(0.0, 200.0),
        help="3D 有效范围 [LOW, HIGH]，X/Y/Z 超出则候选视为无效，默认 0 200",
    )
    ap.add_argument(
        "--use_fish_id",
        action="store_true",
        help="Step1 中按 2D fish_id a/b/c 顺序取点；默认关闭，仅按 score/x 排序",
    )
    ap.add_argument(
        "--no_save_best_json",
        action="store_true",
        help="不额外保存 {set_key}_best.json（只导出 CSV）",
    )
    ap.add_argument(
        "--shift_to_positive",
        action="store_true",
        help="将 3D 坐标整体平移到非负区间（可选）",
    )
    ap.add_argument(
        "--no_cross_frame_tracking",
        action="store_true",
        help="禁用 Step 4 跨帧追踪（仅用 Step 3 每帧最优，ID 可能每帧不同）",
    )
    ap.add_argument(
        "--flow_weight",
        type=float,
        default=10.0,
        help="跨帧追踪时运动方向惩罚权重（mm），默认 10.0",
    )
    ap.add_argument(
        "--no_fill_missing_3d",
        action="store_true",
        help="不补全缺失帧（导出 CSV 可能缺行）；默认会补全全部帧并用三次样条填充",
    )
    args = ap.parse_args()

    bounds: Tuple[float, float] | None = None
    if args.xyz_bounds is not None:
        bounds = (float(args.xyz_bounds[0]), float(args.xyz_bounds[1]))

    result = run_multiview_id_pipeline(
        processed_dir=args.processed_dir,
        set_key=args.set_key,
        model_folder=args.model_folder,
        output_dir=args.output_dir,
        xyz_bounds=bounds,
        use_fish_id=bool(args.use_fish_id),
        save_best_json=not args.no_save_best_json,
        shift_to_positive=bool(args.shift_to_positive),
        cross_frame_tracking=not args.no_cross_frame_tracking,
        flow_weight=float(args.flow_weight),
        fill_missing_3d=not args.no_fill_missing_3d,
    )

    n_frames = len(result["best"])
    print(f"完成三视角 3D ID 分配，共 {n_frames} 帧有有效 a/b/c 结果。")


if __name__ == "__main__":
    main()

