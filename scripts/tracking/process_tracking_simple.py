"""
简化版 2D 跟踪：仅跨帧个体稳定 + 每帧 3 个个体，不做异常值检测与删除。

流程：
1. 加载三视角 SLEAP CSV
2. 跨帧 ID 分配（Hungarian 最小距离匹配，可选运动方向代价）
3. 缺失值填充（确保每帧都有 a / b / c 的坐标，缺的用插值补全）
4. 可选平滑与可视化

不包含：异常值检测、迭代剔除、跳帧缩放阈值等。适合只需要「身份稳定、每帧三人」的轻量流程。
"""
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

from .file_set_generator import generate_file_sets, save_file_sets
from .data_loader import load_set_files, parse_sleap_data
from .id_assignment import assign_ids_temporal
from .missing_value_filling import (
    process_all_fish_missing_values,
    extract_time_from_filename,
)
from .visualization import plot_trajectories_three_views


def process_single_set_simple(
    file_set_info,
    sleap_output_dir,
    output_dir=None,
    num_fish=3,
    max_distance=100,
    flow_weight=0.0,
    fps=166,
    interp_method="linear",
    interp_limit=5,
    smooth=True,
    smooth_window=9,
    smooth_order=2,
    visualize=True,
):
    """
    简化版：仅 ID 分配 + 缺失值填充，无异常值处理。

    Returns:
        dict: tracked_dfs (ID 分配后), final_dfs (填充后), assignments
    """
    set_key = file_set_info.get("key", "unknown")
    print(f"\n处理 set (简化版，无异常值): {set_key}")

    # 1. 加载数据
    print("步骤1: 加载数据...")
    data_dict = load_set_files(file_set_info)
    parsed_data = {}
    for player, df in data_dict.items():
        parsed = parse_sleap_data(df)
        parsed["df"] = df
        parsed_data[player] = parsed

    # 2. 跨帧 ID 分配（Hungarian，保证身份稳定）
    print("步骤2: 分配 ID（跨帧稳定）...")
    fish_ids = [f"fish_{i}" for i in range(num_fish)]
    assignments = assign_ids_temporal(
        parsed_data,
        num_fish=num_fish,
        max_distance=max_distance,
        flow_weight=flow_weight,
    )

    tracked_dfs = {}
    for player in parsed_data.keys():
        df = parsed_data[player]["df"].copy()
        df["fish_id"] = df.apply(
            lambda row: assignments[player].get(row["frame"], {}).get(
                row["track"], "unassigned"
            ),
            axis=1,
        )
        tracked_dfs[player] = df

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print("保存 ID 分配后的 CSV（原始坐标）...")
        for player, df in tracked_dfs.items():
            path = os.path.join(
                output_dir, f"{set_key}_{player}_simple_id_assigned.csv"
            )
            df.to_csv(path, index=False)
            print(f"  已保存: {path}")

    if visualize:
        print("生成 ID 分配后的可视化...")
        viz_path = (
            os.path.join(output_dir, f"{set_key}_simple_after_id.png")
            if output_dir
            else None
        )
        plot_trajectories_three_views(
            tracked_dfs,
            title_suffix="Simple - After ID Assignment",
            save_path=viz_path,
        )

    # 3. 缺失值填充（每帧 3 个个体，缺的插值）
    print("步骤3: 填充缺失值（每帧 3 个个体）...")
    final_dfs = process_all_fish_missing_values(
        tracked_dfs,
        file_set_info=file_set_info,
        fps=fps,
        method=interp_method,
        interp_limit=interp_limit,
        smooth=smooth,
        smooth_window=smooth_window,
        smooth_order=smooth_order,
    )

    if visualize:
        print("生成最终可视化...")
        final_viz_path = (
            os.path.join(output_dir, f"{set_key}_simple_final.png")
            if output_dir
            else None
        )
        plot_trajectories_three_views(
            final_dfs,
            title_suffix="Simple - Final (no outlier removal)",
            save_path=final_viz_path,
        )

    if output_dir:
        for player, df in final_dfs.items():
            path = os.path.join(output_dir, f"{set_key}_{player}_simple_processed.csv")
            df.to_csv(path, index=False)
            print(f"已保存: {path}")

    return {
        "tracked_dfs": tracked_dfs,
        "final_dfs": final_dfs,
        "assignments": assignments,
    }


def process_from_csv_path_simple(
    csv_path,
    sleap_output_dir,
    output_dir=None,
    num_fish=3,
    max_distance=100,
    flow_weight=0.0,
    fps=166,
    interp_method="linear",
    interp_limit=5,
    smooth=True,
    visualize=True,
    **kwargs,
):
    """
    从一条 PLAYER1 的 CSV 路径出发，自动找同 set 的 PLAYER2/3，执行简化版跟踪。
    """
    from .file_set_generator import extract_set_key_from_filename, save_file_sets

    set_key = extract_set_key_from_filename(csv_path)
    if not set_key:
        raise ValueError(f"无法从文件名提取 set key: {csv_path}")

    sleap_output_path = Path(sleap_output_dir)
    top_dir = sleap_output_path / "top_mouth_180"
    side_dir = sleap_output_path / "side_mouth_593"
    if not top_dir.exists() or not side_dir.exists():
        inferred = _infer_sleap_output_dir_from_csv(csv_path)
        if inferred:
            sleap_output_dir = inferred
            sleap_output_path = Path(sleap_output_dir)
            top_dir = sleap_output_path / "top_mouth_180"
            side_dir = sleap_output_path / "side_mouth_593"
        if not top_dir.exists() or not side_dir.exists():
            raise ValueError(
                f"目录不存在: {top_dir} 或 {side_dir}。"
                "请确认 SLEAP_OUTPUT_DIR 并先运行路径配置。"
            )

    file_sets_json_path = os.path.join(sleap_output_dir, "file_sets.json")
    if not os.path.exists(file_sets_json_path):
        file_sets = generate_file_sets(sleap_output_dir)
        save_file_sets(file_sets, file_sets_json_path)
    else:
        with open(file_sets_json_path, "r", encoding="utf-8") as f:
            file_sets = json.load(f)

    if set_key not in file_sets:
        file_sets = generate_file_sets(sleap_output_dir)
        save_file_sets(file_sets, file_sets_json_path)
    if set_key not in file_sets:
        raise ValueError(f"未找到 set key: {set_key}")

    file_set_info = file_sets[set_key]
    return process_single_set_simple(
        file_set_info,
        sleap_output_dir,
        output_dir=output_dir,
        num_fish=num_fish,
        max_distance=max_distance,
        flow_weight=flow_weight,
        fps=fps,
        interp_method=interp_method,
        interp_limit=interp_limit,
        smooth=smooth,
        visualize=visualize,
        **kwargs,
    )


def process_from_file_set_key_simple(
    file_set_key,
    file_sets_json_path,
    sleap_output_dir,
    output_dir=None,
    **kwargs,
):
    """
    根据 file_set_key 处理数据（简化版）。
    """
    with open(file_sets_json_path, "r", encoding="utf-8") as f:
        file_sets = json.load(f)

    if file_set_key not in file_sets:
        raise ValueError(f"未找到 set key: {file_set_key}")

    file_set_info = file_sets[file_set_key]
    return process_single_set_simple(
        file_set_info,
        sleap_output_dir,
        output_dir=output_dir,
        **kwargs,
    )


def _infer_sleap_output_dir_from_csv(csv_path):
    """从 CSV 路径向上查找包含 top_mouth_* 与 side_mouth_* 的目录。"""
    p = Path(csv_path).resolve()
    for _ in range(10):
        if p.parent == p:
            break
        p = p.parent
        try:
            generate_file_sets(str(p))
            return str(p)
        except (ValueError, OSError):
            continue
    return None
