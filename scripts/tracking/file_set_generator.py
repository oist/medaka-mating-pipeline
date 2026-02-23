"""
自动生成file_sets.json文件，将同一实验的3个视角文件分组
"""
import os
import json
import re
from pathlib import Path
from collections import defaultdict


def extract_set_key_from_filename(filename):
    """
    从文件名中提取 set key，保证同一 clip 的 PLAYER1/2/3 得到相同 key。
    支持重复前缀（如 20251225_K_1_2_clips_20251225_K_1_2_FCMC_...）：取「最后一个」日期段作为前缀，避免 key 不一致。
    例: 20251225_K_1_2_clips_20251225_K_1_2_FCMC_PLAYER1_m1tf_Courtship_290.729-296.331.csv
     -> 20251225_K_1_2_FCMC_m1tf_Courtship_290.729-296.331
    """
    basename = os.path.basename(filename)
    basename = basename.replace(".csv", "")

    pattern = r"_PLAYER\d+_"
    match = re.search(pattern, basename)
    if not match:
        return None

    before_player = basename[: match.start()]
    after_player = basename[match.end() :]

    # 找「最后一个」8位日期（YYYYMMDD）开头的一段，避免 20251225_K_1_2_clips_20251225_K_1_2_FCMC 取到整段
    date_parts = list(re.finditer(r"\d{8}(?:_[^_]+)*", before_player))
    if date_parts:
        prefix = date_parts[-1].group(0)  # 最后一段，如 20251225_K_1_2_FCMC
        return f"{prefix}_{after_player}"
    # 兼容：整段里没有明显日期段时，用「最后一个」8位数字到结尾
    last_8 = list(re.finditer(r"\d{8}", before_player))
    if last_8:
        start = last_8[-1].start()
        prefix = before_player[start:]
        return f"{prefix}_{after_player}"
    return f"{before_player}_{after_player}"


def generate_file_sets(sleap_output_dir):
    """
    自动扫描sleap_output目录，生成file_sets.json
    
    Args:
        sleap_output_dir: sleap输出目录路径
        
    Returns:
        dict: file_sets字典
    """
    sleap_output_path = Path(sleap_output_dir)
    
    # 查找所有CSV文件
    top_dir = sleap_output_path / "top_mouth_180"
    side_dir = sleap_output_path / "side_mouth_593"
    
    if not top_dir.exists() or not side_dir.exists():
        raise ValueError(f"目录不存在: {top_dir} 或 {side_dir}")
    
    # 按set key分组
    sets_dict = defaultdict(dict)
    
    # 扫描 top_mouth_180 目录及子目录 (PLAYER1)，支持 top_mouth_180/xxx_clips/*.csv
    for csv_file in top_dir.rglob("*.csv"):
        set_key = extract_set_key_from_filename(str(csv_file))
        if set_key:
            sets_dict[set_key]["key"] = set_key
            sets_dict[set_key]["player1_file"] = str(csv_file.resolve())
    
    # 扫描 side_mouth_593 目录及子目录 (PLAYER2和PLAYER3)
    for csv_file in side_dir.rglob("*.csv"):
        set_key = extract_set_key_from_filename(str(csv_file))
        if set_key:
            if "PLAYER2" in csv_file.name:
                sets_dict[set_key]["player2_file"] = str(csv_file.resolve())
            elif "PLAYER3" in csv_file.name:
                sets_dict[set_key]["player3_file"] = str(csv_file.resolve())
    
    # 只保留有3个文件的set
    complete_sets = {}
    for key, files in sets_dict.items():
        if "player1_file" in files and "player2_file" in files and "player3_file" in files:
            complete_sets[key] = files
    
    return complete_sets


def save_file_sets(file_sets, output_path):
    """保存file_sets到JSON文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(file_sets, f, indent=2, ensure_ascii=False)
    print(f"已保存 {len(file_sets)} 个set到 {output_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python file_set_generator.py <sleap_output_dir> [output_json_path]")
        sys.exit(1)
    
    sleap_dir = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(sleap_dir, "file_sets.json")
    
    file_sets = generate_file_sets(sleap_dir)
    save_file_sets(file_sets, output_path)
