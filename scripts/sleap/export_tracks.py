from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import pandas as pd

try:
    import sleap  # type: ignore[import]
except ImportError:
    sleap = None  # type: ignore[assignment]


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _sleap_cmd(exe: str, args: list[str]) -> list[str]:
    """Use executable from PATH if present; otherwise return [exe, *args] and let the system report error."""
    if shutil.which(exe) is not None:
        return [exe, *args]
    return [exe, *args]


def extract_video_and_model_info(slp_path: Path):
    """
    从SLEAP文件中提取视频文件名和模型名。
    
    返回:
    - video_name: 视频文件名（不含扩展名）
    - model_name: 模型名（从slp文件路径推断，例如从目录名）
    """
    if sleap is None:
        raise SystemExit(
            "Cannot import sleap package. Please install SLEAP first, e.g.:\n"
            "pip install sleap or conda install -c conda-forge sleap"
        )
    
    labels = None
    last_error = None
    
    # Try different loading methods
    try:
        from sleap.io.dataset import Labels  # type: ignore[import]
        labels = Labels.load_file(str(slp_path))  # type: ignore[attr-defined]
    except Exception as e:  # noqa: BLE001
        last_error = e
        try:
            labels = sleap.load_file(str(slp_path))  # type: ignore[attr-defined]
        except Exception as e2:  # noqa: BLE001
            last_error = e2
            try:
                labels = sleap.Labels.load_file(str(slp_path))  # type: ignore[attr-defined]
            except Exception as e3:  # noqa: BLE001
                last_error = e3
                try:
                    import sleap_io as sio  # type: ignore[import]
                    labels = sio.load_slp(str(slp_path))  # type: ignore[attr-defined]
                except Exception:  # noqa: BLE001
                    pass
    
    if labels is None:
        error_msg = (
            f"Failed to load SLEAP file: {slp_path}\n"
            f"Error: {type(last_error).__name__}: {last_error}\n"
            "Please check:\n"
            "1. SLEAP is correctly installed\n"
            "2. File path is correct\n"
            "3. File format is valid"
        )
        raise SystemExit(error_msg)
    
    # 提取视频文件名
    video_name = None
    if hasattr(labels, 'videos') and len(labels.videos) > 0:
        video_path = Path(labels.videos[0].filename)
        video_name = video_path.stem  # 不含扩展名的文件名
    
    # 从slp文件路径推断模型名
    # 通常模型名在目录名中，例如: D:\sleap_output\top_mouth_180\xxx.slp
    # 模型名可能是 "top_mouth_180"
    model_name = None
    slp_parent = slp_path.parent
    if slp_parent.name:  # 如果父目录有名称
        model_name = slp_parent.name
    
    return video_name, model_name


def load_slp_data_all_tracks(slp_path: Path):
    """Load all tracks data from .slp file"""
    if sleap is None:
        raise SystemExit(
            "Cannot import sleap package. Please install SLEAP first, e.g.:\n"
            "pip install sleap or conda install -c conda-forge sleap"
        )
    
    print(f"Loading SLEAP file: {slp_path}")
    
    labels = None
    last_error = None
    
    # Try different loading methods
    try:
        from sleap.io.dataset import Labels  # type: ignore[import]
        labels = Labels.load_file(str(slp_path))  # type: ignore[attr-defined]
    except Exception as e:  # noqa: BLE001
        last_error = e
        try:
            labels = sleap.load_file(str(slp_path))  # type: ignore[attr-defined]
        except Exception as e2:  # noqa: BLE001
            last_error = e2
            try:
                labels = sleap.Labels.load_file(str(slp_path))  # type: ignore[attr-defined]
            except Exception as e3:  # noqa: BLE001
                last_error = e3
                try:
                    import sleap_io as sio  # type: ignore[import]
                    labels = sio.load_slp(str(slp_path))  # type: ignore[attr-defined]
                except Exception:  # noqa: BLE001
                    pass
    
    if labels is None:
        error_msg = (
            f"Failed to load SLEAP file: {slp_path}\n"
            f"Error: {type(last_error).__name__}: {last_error}\n"
            "Please check:\n"
            "1. SLEAP is correctly installed\n"
            "2. File path is correct\n"
            "3. File format is valid"
        )
        raise SystemExit(error_msg)

    rows: list[dict] = []

    # Iterate over all labeled frames
    for lf in labels.labeled_frames:  # type: ignore[attr-defined]
        frame_idx = lf.frame_idx

        for inst in lf.instances:
            # Get track name
            track_name = None
            if getattr(inst, "track", None) is not None:
                track_name = getattr(inst.track, "name", None) or getattr(
                    inst.track, "track_id", None
                )
            if track_name is None:
                track_name = f"untracked_{id(inst)}"

            # Collect all keypoints for this instance
            points_data = inst.points
            
            # Method 1: If dict (old SLEAP version)
            if isinstance(points_data, dict) or (hasattr(points_data, 'items') and callable(getattr(points_data, 'items'))):
                try:
                    for node, point in points_data.items():
                        score = float(getattr(point, "score", 1.0))
                        rows.append({
                            "frame": int(frame_idx),
                            "track": str(track_name),
                            "node": str(node),
                            "x": float(point.x),
                            "y": float(point.y),
                            "score": score,
                        })
                except (AttributeError, TypeError):
                    pass
            
            # Method 2: Try numpy array (new SLEAP version)
            else:
                try:
                    import numpy as np
                    if hasattr(inst, "numpy") or isinstance(points_data, np.ndarray):
                        # Get node names
                        skeleton = getattr(labels, "skeleton", None)
                        if skeleton is None and hasattr(lf, "video"):
                            video = lf.video
                            if hasattr(video, "skeleton"):
                                skeleton = video.skeleton
                        
                        node_names = []
                        if skeleton is not None:
                            if hasattr(skeleton, "nodes"):
                                node_names = [str(node) for node in skeleton.nodes]
                            elif hasattr(skeleton, "node_names"):
                                node_names = [str(name) for name in skeleton.node_names]
                        
                        # Get point data
                        if hasattr(inst, "numpy"):
                            points_array = inst.numpy()  # type: ignore[attr-defined]
                        else:
                            points_array = points_data
                        
                        if isinstance(points_array, np.ndarray) and len(points_array.shape) >= 2:
                            num_nodes = points_array.shape[0]
                            for i in range(num_nodes):
                                node_name = node_names[i] if i < len(node_names) else f"node_{i}"
                                x = float(points_array[i, 0])
                                y = float(points_array[i, 1])
                                score = float(points_array[i, 2]) if points_array.shape[1] > 2 else 1.0
                                
                                rows.append({
                                    "frame": int(frame_idx),
                                    "track": str(track_name),
                                    "node": str(node_name),
                                    "x": x,
                                    "y": y,
                                    "score": score,
                                })
                except (ImportError, AttributeError, TypeError, IndexError):
                    pass

    if not rows:
        raise SystemExit("Error: No tracking data found in .slp file.")

    df = pd.DataFrame.from_records(rows)
    print(f"Loaded {len(df)} data points, {df['track'].nunique()} tracks")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export SLEAP predictions (.slp) to analysis-friendly formats via `sleap-convert`. "
            "(SLEAP 1.5.x uses `sleap-convert`; `sleap-export` is legacy.)\n"
            "For CSV format, use --all-tracks to export all tracks (not just the first one)."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input predictions .slp (or labels .slp).",
    )
    parser.add_argument(
        "--format",
        default="analysis",
        choices=["analysis", "analysis.csv",
                 "analysis.nix", "h5", "json", "slp"],
        help="Output format passed to `sleap-convert --format`.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path. If not provided and using --all-tracks with CSV format, "
             "output filename will be auto-generated as: {model_name}_{video_name}.csv",
    )
    parser.add_argument(
        "--video",
        help="Optional video path for conversions that require a video reference.",
    )
    parser.add_argument(
        "--all-tracks",
        action="store_true",
        help="For CSV format, export all tracks instead of just the first one. "
             "This uses direct SLEAP API instead of sleap-convert.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    # If CSV format and need all tracks, use direct loading
    if args.format == "analysis.csv" and args.all_tracks:
        print("Exporting all tracks to CSV using direct loading...")
        df = load_slp_data_all_tracks(input_path)
        
        # 提取视频名和模型名
        video_name, model_name = extract_video_and_model_info(input_path)
        
        if video_name is None:
            raise SystemExit(
                "无法从SLEAP文件中提取视频名。请使用 --output 参数手动指定输出路径。"
            )
        
        if model_name is None:
            model_name = "unknown_model"
            print(f"警告: 无法从路径推断模型名，使用默认值: {model_name}")
        
        # 生成输出文件名: {model_name}_{video_name}.csv
        output_filename = f"{model_name}_{video_name}.csv"
        
        # 确定输出路径
        if args.output is None:
            # 如果未指定输出路径，使用输入文件所在目录
            output_path = input_path.parent / output_filename
            print(f"自动生成输出文件名: {output_filename}")
        else:
            output_path = Path(args.output)
            # 如果指定的输出路径是目录，则在目录中生成自动命名的文件
            if output_path.is_dir() or (not output_path.suffix and not output_path.exists()):
                output_path = output_path / output_filename
                print(f"在指定目录中生成文件: {output_filename}")
            else:
                # 如果指定的是文件路径，使用该路径（但提示用户可以使用自动命名）
                print(f"使用指定的输出路径: {output_path}")
                print(f"提示: 如果希望使用自动命名 ({output_filename})，可以指定目录而不是文件路径")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Successfully exported to: {output_path}")
        print(f"  - Total data points: {len(df)}")
        print(f"  - Number of tracks: {df['track'].nunique()}")
        print(f"  - Number of nodes: {df['node'].nunique()}")
        print(f"  - Frame range: {df['frame'].min()} - {df['frame'].max()}")
        return

    # Otherwise use sleap-convert
    if args.output is None:
        raise SystemExit("--output 参数是必需的（除非使用 --all-tracks 和 CSV 格式）")
    
    convert_args: list[str] = [str(input_path), "--format", args.format]
    if args.output:
        convert_args += ["--output", args.output]
    if args.video:
        convert_args += ["--video", args.video]

    _run(_sleap_cmd("sleap-convert", convert_args))


if __name__ == "__main__":
    main()
