"""
主处理脚本：整合所有步骤
"""
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

from .file_set_generator import generate_file_sets, save_file_sets
from .data_loader import load_set_files, parse_sleap_data
from .id_assignment import assign_ids_temporal
from .outlier_detection import process_all_fish_outliers
from .missing_value_filling import process_all_fish_missing_values, extract_time_from_filename
from .visualization import plot_trajectories_three_views


def process_single_set(file_set_info, sleap_output_dir, output_dir=None,
                       max_distance=100, max_jump_distance=200, fps=166,
                       interp_method='linear', interp_limit=None, smooth=True,
                       visualize=True, max_iterations=10,
                       use_frame_gap_scaling=True, max_jump_per_frame=100,
                       flow_weight=0, **kwargs):
    """
    处理单个set的数据
    
    Args:
        file_set_info: dict包含player1_file, player2_file, player3_file
        sleap_output_dir: sleap输出目录
        output_dir: 输出目录
        max_distance: ID分配的最大距离阈值
        outlier_method: 异常值检测方法
        interp_method: 插值方法
        interp_limit: 最大连续缺失值数量
        smooth: 是否平滑
        visualize: 是否生成可视化
        
    Returns:
        dict: 处理结果
    """
    print(f"\n处理set: {file_set_info.get('key', 'unknown')}")
    
    # 步骤1: 加载数据
    print("步骤1: 加载数据...")
    data_dict = load_set_files(file_set_info)
    
    # 解析数据
    parsed_data = {}
    for player, df in data_dict.items():
        parsed = parse_sleap_data(df)
        parsed['df'] = df  # 保留原始DataFrame
        parsed_data[player] = parsed
    
    # 步骤2: ID分配（Hungarian + 可选运动方向代价）
    print("步骤2: 分配ID...")
    assignments = assign_ids_temporal(
        parsed_data, num_fish=3, max_distance=max_distance, flow_weight=flow_weight
    )
    
    # 创建带ID的DataFrame
    tracked_dfs = {}
    for player in parsed_data.keys():
        df = parsed_data[player]['df']
        df = df.copy()
        df['fish_id'] = df.apply(
            lambda row: assignments[player].get(row['frame'], {}).get(row['track'], 'unassigned'),
            axis=1
        )
        tracked_dfs[player] = df
    
    # 保存ID分配后的CSV文件（原始坐标，只分配了ID）
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        set_key = file_set_info.get('key', 'unknown')
        print("保存ID分配后的CSV文件（原始坐标）...")
        for player, df in tracked_dfs.items():
            id_assigned_file = os.path.join(output_dir, f"{set_key}_{player}_id_assigned.csv")
            df.to_csv(id_assigned_file, index=False)
            print(f"  已保存: {id_assigned_file}")
    
    # 可视化ID分配后的结果
    if visualize:
        print("生成ID分配后的可视化...")
        set_key = file_set_info.get('key', 'unknown')
        id_viz_path = os.path.join(output_dir, f"{set_key}_after_id_assignment.png") if output_dir else None
        plot_trajectories_three_views(tracked_dfs, title_suffix="After ID Assignment", save_path=id_viz_path)
    
    # 步骤3: 异常值处理（可选：1帧>100px, 2帧>200px... 或固定阈值）
    print("步骤3: 处理异常值（迭代直到没有异常值）...")
    cleaned_dfs = process_all_fish_outliers(
        tracked_dfs,
        max_jump_distance=max_jump_distance,
        iterative=True,
        max_iterations=kwargs.get('max_iterations', max_iterations),
        use_frame_gap_scaling=use_frame_gap_scaling,
        max_jump_per_frame=max_jump_per_frame,
    )
    
    # 步骤4: 缺失值填充
    print("步骤4: 填充缺失值...")
    # 逻辑：用上下两帧的数据进行插值补齐，确保每一帧都有3个个体的数据
    final_dfs = process_all_fish_missing_values(
        cleaned_dfs,
        file_set_info=file_set_info,  # 传递文件信息以提取时间范围
        fps=fps,  # 166fps
        method=interp_method, 
        interp_limit=interp_limit, 
        smooth=smooth,
        smooth_window=9,
        smooth_order=2
    )
    
    # 可视化最终结果
    if visualize:
        print("生成最终处理后的可视化...")
        set_key = file_set_info.get('key', 'unknown')
        final_viz_path = os.path.join(output_dir, f"{set_key}_final.png") if output_dir else None
        plot_trajectories_three_views(final_dfs, title_suffix="Final Processed", save_path=final_viz_path)
    
    # 保存结果
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        set_key = file_set_info.get('key', 'unknown')
        
        for player, df in final_dfs.items():
            output_file = os.path.join(output_dir, f"{set_key}_{player}_processed.csv")
            df.to_csv(output_file, index=False)
            print(f"已保存: {output_file}")
    
    return {
        'tracked_dfs': tracked_dfs,
        'final_dfs': final_dfs,
        'assignments': assignments
    }


def process_from_file_set_key(file_set_key, file_sets_json_path, sleap_output_dir, 
                              output_dir=None, **kwargs):
    """
    根据file_set_key处理数据
    
    Args:
        file_set_key: set的key
        file_sets_json_path: file_sets.json路径
        sleap_output_dir: sleap输出目录
        output_dir: 输出目录
        **kwargs: 其他处理参数
        
    Returns:
        dict: 处理结果
    """
    # 加载file_sets.json
    with open(file_sets_json_path, 'r', encoding='utf-8') as f:
        file_sets = json.load(f)
    
    if file_set_key not in file_sets:
        raise ValueError(f"未找到set key: {file_set_key}")
    
    file_set_info = file_sets[file_set_key]
    return process_single_set(file_set_info, sleap_output_dir, output_dir=output_dir, **kwargs)


def _infer_sleap_output_dir_from_csv(csv_path):
    """从 CSV 路径向上查找包含 top_mouth_* 与 side_mouth_* 的目录作为 sleap 根目录。"""
    from .file_set_generator import generate_file_sets
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


def process_from_csv_path(csv_path, sleap_output_dir, output_dir=None, **kwargs):
    """
    根据CSV文件路径处理数据（自动查找对应的set）
    
    Args:
        csv_path: CSV文件路径
        sleap_output_dir: sleap输出目录（若该目录下没有 top_mouth_180 / side_mouth_593，会尝试从 csv_path 自动推断）
        output_dir: 输出目录
        **kwargs: 其他处理参数
        
    Returns:
        dict: 处理结果
    """
    from .file_set_generator import extract_set_key_from_filename, generate_file_sets, save_file_sets

    set_key = extract_set_key_from_filename(csv_path)
    if not set_key:
        raise ValueError(f"无法从文件名提取set key: {csv_path}")

    sleap_output_path = Path(sleap_output_dir)
    top_dir = sleap_output_path / "top_mouth_180"
    side_dir = sleap_output_path / "side_mouth_593"
    if not top_dir.exists() or not side_dir.exists():
        inferred = _infer_sleap_output_dir_from_csv(csv_path)
        if inferred:
            print(f"已从 CSV 路径推断 SLEAP 根目录: {inferred}")
            sleap_output_dir = inferred
        else:
            raise ValueError(
                f"目录不存在: {top_dir} 或 {side_dir}。"
                "请确认「路径与配置」中 SLEAP_OUTPUT_DIR 指向实际导出目录（如 D:\\sleap_output），并先运行该 cell。"
            )

    file_sets_json_path = os.path.join(sleap_output_dir, "file_sets.json")
    if not os.path.exists(file_sets_json_path):
        print("生成file_sets.json...")
        file_sets = generate_file_sets(sleap_output_dir)
        save_file_sets(file_sets, file_sets_json_path)
    else:
        with open(file_sets_json_path, 'r', encoding='utf-8') as f:
            file_sets = json.load(f)

    if set_key not in file_sets:
        print(f"未找到set key: {set_key}，重新生成file_sets.json...")
        file_sets = generate_file_sets(sleap_output_dir)
        save_file_sets(file_sets, file_sets_json_path)

    if set_key not in file_sets:
        raise ValueError(f"未找到set key: {set_key}，请检查文件名格式是否正确")

    file_set_info = file_sets[set_key]
    return process_single_set(file_set_info, sleap_output_dir, output_dir=output_dir, **kwargs)
