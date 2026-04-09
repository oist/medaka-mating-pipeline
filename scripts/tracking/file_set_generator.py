"""Build `file_sets.json` by grouping the three camera views for each clip."""
import os
import json
import re
from pathlib import Path
from collections import defaultdict


def extract_set_key_from_filename(filename):
    """
    Extract a stable set key from a filename so PLAYER1/2/3 from the same clip
    collapse to the same key.

    Repeated prefixes are supported. For names like
    `20251225_K_1_2_clips_20251225_K_1_2_FCMC_...`, the final date block is
    used as the prefix to avoid mismatched keys.

    Example:
      20251225_K_1_2_clips_20251225_K_1_2_FCMC_PLAYER1_m1tf_Courtship_290.729-296.331.csv
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

    # Use the final 8-digit date prefix to avoid keeping the duplicated
    # `..._clips_...` prefix in the key.
    date_parts = list(re.finditer(r"\d{8}(?:_[^_]+)*", before_player))
    if date_parts:
        prefix = date_parts[-1].group(0)
        return f"{prefix}_{after_player}"
    # Fallback: if there is no obvious date block, use the final 8-digit number.
    last_8 = list(re.finditer(r"\d{8}", before_player))
    if last_8:
        start = last_8[-1].start()
        prefix = before_player[start:]
        return f"{prefix}_{after_player}"
    return f"{before_player}_{after_player}"


def generate_file_sets(sleap_output_dir):
    """
    Scan the SLEAP export root and build the `file_sets` mapping.

    Args:
        sleap_output_dir: Root path containing the exported CSV files.

    Returns:
        Dictionary ready to be written to `file_sets.json`.
    """
    sleap_output_path = Path(sleap_output_dir)
    
    # Locate the expected top and side export directories.
    top_dir = sleap_output_path / "top_mouth_180"
    side_dir = sleap_output_path / "side_mouth_593"
    
    if not top_dir.exists() or not side_dir.exists():
        raise ValueError(f"Directory does not exist: {top_dir} or {side_dir}")
    
    # Group files by the inferred set key.
    sets_dict = defaultdict(dict)
    
    # Scan PLAYER1 files under the top-view export directory.
    for csv_file in top_dir.rglob("*.csv"):
        set_key = extract_set_key_from_filename(str(csv_file))
        if set_key:
            sets_dict[set_key]["key"] = set_key
            sets_dict[set_key]["player1_file"] = str(csv_file.resolve())
    
    # Scan PLAYER2 and PLAYER3 files under the side-view export directory.
    for csv_file in side_dir.rglob("*.csv"):
        set_key = extract_set_key_from_filename(str(csv_file))
        if set_key:
            if "PLAYER2" in csv_file.name:
                sets_dict[set_key]["player2_file"] = str(csv_file.resolve())
            elif "PLAYER3" in csv_file.name:
                sets_dict[set_key]["player3_file"] = str(csv_file.resolve())
    
    # Keep only complete sets with all three views present.
    complete_sets = {}
    for key, files in sets_dict.items():
        if "player1_file" in files and "player2_file" in files and "player3_file" in files:
            complete_sets[key] = files
    
    return complete_sets


def save_file_sets(file_sets, output_path):
    """Write `file_sets` to a JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(file_sets, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(file_sets)} set(s) to {output_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python file_set_generator.py <sleap_output_dir> [output_json_path]")
        sys.exit(1)
    
    sleap_dir = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(sleap_dir, "file_sets.json")
    
    file_sets = generate_file_sets(sleap_dir)
    save_file_sets(file_sets, output_path)
