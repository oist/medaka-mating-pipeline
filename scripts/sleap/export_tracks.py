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
    """从 .slp 导出所有轨迹，track 使用与 overlay 视频一致的稳定 ID（Track_0/1/2 或 track name）。"""
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

    # 与 overlay 视频一致的稳定 track 列表（sleap-convert / 视频渲染用同一套）
    label_tracks = getattr(labels, "tracks", None) or []
    try:
        label_tracks = list(label_tracks)
    except Exception:
        label_tracks = []

    # 与 03_sleap.ipynb 里的 _instance_points 保持同一逻辑（只是在这里顺便保留 score）
    def _instance_points_with_score(inst, lf, score_thresh: float = 0.01) -> dict[str, tuple[float, float, float]]:
        """
        返回: node_name -> (x, y, score)
        - 坐标提取方式与 Notebook 中生成 overlay 的 `_instance_points` 完全一致；
        - 仅在 score >= score_thresh 时保留该点。
        """
        pts = getattr(inst, "points", None)
        if pts is None:
            return {}

        out: dict[str, tuple[float, float, float]] = {}

        # 情况 1：老版本，points 是 dict-like（value 有 x/y/score 属性）
        if hasattr(pts, "items"):
            for node, p in pts.items():
                score = float(getattr(p, "score", 1.0))
                if score < score_thresh:
                    continue
                x = float(getattr(p, "x", p[0] if hasattr(p, "__getitem__") else 0.0))
                y = float(getattr(p, "y", p[1] if hasattr(p, "__getitem__") else 0.0))
                out[str(node)] = (x, y, score)
            return out

        # 情况 2：新版本，points 为 numpy 数组，需结合 skeleton 解析节点名
        try:
            import numpy as np  # 局部导入，避免环境里没装 numpy 时整个脚本直接 ImportError
        except ImportError as e:  # noqa: BLE001
            raise SystemExit(
                "numpy is required to export all tracks. "
                "Please install it, e.g.:\n  pip install numpy"
            ) from e
        sk = getattr(labels, "skeleton", None)
        if sk is None and hasattr(lf, "video"):
            sk = getattr(lf.video, "skeleton", None)

        node_names: list[str] = []
        if sk is not None:
            node_names = getattr(sk, "node_names", None) or getattr(sk, "nodes", [])
            node_names = [str(n) for n in node_names]

        points_array = getattr(inst, "numpy", lambda: None)() if hasattr(inst, "numpy") else pts
        if isinstance(points_array, np.ndarray) and len(points_array.shape) >= 2:
            for i in range(points_array.shape[0]):
                name = node_names[i] if i < len(node_names) else f"node_{i}"
                x = float(points_array[i, 0])
                y = float(points_array[i, 1])
                score = float(points_array[i, 2]) if points_array.shape[1] > 2 else 1.0
                if score < score_thresh:
                    continue
                out[name] = (x, y, score)

        return out

    rows: list[dict] = []

    # 遍历所有帧与 instance，使用与 overlay 一致的点解析与置信度阈值
    for lf in labels.labeled_frames:  # type: ignore[attr-defined]
        frame_idx = int(getattr(lf, "frame_idx", 0))

        for inst_idx, inst in enumerate(getattr(lf, "instances", [])):
            # 使用与 overlay 一致的稳定 track 身份（不用 id(inst)，否则每帧都是新 ID）
            track_name = None
            if getattr(inst, "track", None) is not None:
                track_name = getattr(inst.track, "name", None) or getattr(
                    inst.track, "track_id", None
                )
                if track_name is None and label_tracks:
                    try:
                        idx = label_tracks.index(inst.track)
                        track_name = f"Track_{idx}"
                    except (ValueError, TypeError):
                        pass
            if track_name is None:
                # 无 track 时用「帧号 + 帧内序号」，保证至少同帧内顺序稳定
                track_name = f"untracked_f{frame_idx}_{inst_idx}"

            pts = _instance_points_with_score(inst, lf)
            if not pts:
                continue

            for node_name, (x, y, score) in pts.items():
                rows.append(
                    {
                        "frame": frame_idx,
                        "track": str(track_name),
                        "node": str(node_name),
                        "x": float(x),
                        "y": float(y),
                        "score": float(score),
                    }
                )

    if not rows:
        raise SystemExit("Error: No tracking data found in .slp file.")

    df = pd.DataFrame.from_records(rows)
    print(f"Loaded {len(df)} data points, {df['track'].nunique()} tracks")
    return df


def load_analysis_h5_all_tracks(h5_path: Path) -> pd.DataFrame:
    """
    从官方 sleap-convert 导出的 analysis HDF5 中读取所有轨迹，展开为长表 CSV：
    列为 frame, track, node, x, y, score。
    """
    try:
        import h5py  # type: ignore[import]
        import numpy as np  # type: ignore[import]
    except ImportError as e:  # noqa: BLE001
        raise SystemExit(
            "h5py and numpy are required to read analysis HDF5.\n"
            "Please install them, e.g.:\n"
            "  pip install h5py numpy"
        ) from e

    if not h5_path.exists():
        raise SystemExit(f"Analysis HDF5 not found: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        if "tracks" not in f:
            raise SystemExit(f"'tracks' dataset not found in analysis HDF5: {h5_path}")
        tracks_ds = f["tracks"]
        tracks = tracks_ds[()]  # 4D array

        # 轨迹与节点名称
        track_names_ds = f.get("track_names")
        if track_names_ds is not None:
            raw = track_names_ds[()]
            track_names = [
                (name.decode("utf-8") if isinstance(name, (bytes, bytearray)) else str(name))
                for name in raw
            ]
        else:
            track_names = []

        node_names_ds = f.get("node_names")
        if node_names_ds is not None:
            raw_nodes = node_names_ds[()]
            node_names = [
                (name.decode("utf-8") if isinstance(name, (bytes, bytearray)) else str(name))
                for name in raw_nodes
            ]
        else:
            node_names = []

        point_scores_ds = f.get("point_scores")
        point_scores = point_scores_ds[()] if point_scores_ds is not None else None

    if tracks.ndim != 4:
        raise SystemExit(f"Unsupported tracks ndim={tracks.ndim} in analysis HDF5 (expected 4).")

    # 推断维度顺序：
    # - 默认 sleap-convert 使用 MATLAB preset：tracks: (n_tracks, 2, n_nodes, n_frames)
    # - 若最后一维为 2，则视为标准 preset：tracks: (n_frames, n_tracks, n_nodes, 2)
    use_matlab_order = False
    if tracks.shape[1] == 2:
        use_matlab_order = True
    elif tracks.shape[-1] == 2:
        use_matlab_order = False
    else:
        # 回退：优先按 MATLAB 解释
        use_matlab_order = True

    rows: list[dict] = []

    if use_matlab_order:
        # (n_tracks, 2, n_nodes, n_frames)
        n_tracks, _, n_nodes, n_frames = tracks.shape
        if not track_names:
            track_names = [f"Track_{i}" for i in range(n_tracks)]
        if not node_names:
            node_names = [f"node_{i}" for i in range(n_nodes)]

        if point_scores is not None and point_scores.shape != (n_tracks, n_nodes, n_frames):
            point_scores = None

        for t in range(n_tracks):
            tname = track_names[t] if t < len(track_names) else f"Track_{t}"
            for n in range(n_nodes):
                nname = node_names[n] if n < len(node_names) else f"node_{n}"
                for fi in range(n_frames):
                    x = float(tracks[t, 0, n, fi])
                    y = float(tracks[t, 1, n, fi])
                    if not (np.isfinite(x) and np.isfinite(y)):
                        continue
                    score = 1.0
                    if point_scores is not None:
                        score = float(point_scores[t, n, fi])
                    rows.append(
                        {
                            "frame": int(fi),
                            "track": str(tname),
                            "node": str(nname),
                            "x": x,
                            "y": y,
                            "score": score,
                        }
                    )
    else:
        # 标准 preset：tracks: (n_frames, n_tracks, n_nodes, 2)
        n_frames, n_tracks, n_nodes, _ = tracks.shape
        if not track_names:
            track_names = [f"Track_{i}" for i in range(n_tracks)]
        if not node_names:
            node_names = [f"node_{i}" for i in range(n_nodes)]

        if point_scores is not None and point_scores.shape != (n_frames, n_tracks, n_nodes):
            point_scores = None

        for fi in range(n_frames):
            for t in range(n_tracks):
                tname = track_names[t] if t < len(track_names) else f"Track_{t}"
                for n in range(n_nodes):
                    nname = node_names[n] if n < len(node_names) else f"node_{n}"
                    x = float(tracks[fi, t, n, 0])
                    y = float(tracks[fi, t, n, 1])
                    if not (np.isfinite(x) and np.isfinite(y)):
                        continue
                    score = 1.0
                    if point_scores is not None:
                        score = float(point_scores[fi, t, n])
                    rows.append(
                        {
                            "frame": int(fi),
                            "track": str(tname),
                            "node": str(nname),
                            "x": x,
                            "y": y,
                            "score": score,
                        }
                    )

    if not rows:
        raise SystemExit(f"No valid tracking data found in analysis HDF5: {h5_path}")

    df = pd.DataFrame.from_records(rows)
    print(
        f"Loaded {len(df)} data points from analysis HDF5, "
        f"{df['track'].nunique()} tracks, {df['node'].nunique()} nodes"
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export SLEAP predictions (.slp) to analysis-friendly formats via `sleap-convert`. "
            "(SLEAP 1.5.x uses `sleap-convert`; `sleap-export` is legacy.)\n"
            "For CSV with same track identity as overlay video: use --all-tracks (uses stable Track_0/1/2 from .slp)."
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

    # If CSV format and need all tracks, use direct loading from .slp
    if args.format == "analysis.csv" and args.all_tracks:
        print("Exporting all tracks to CSV using direct loading from .slp (all instances per frame)...")
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
