

from __future__ import annotations

"""
批量 SLEAP 跟踪脚本
==================

功能：
- 接受多个 top 视角视频
- 自动使用 top 模型对 top 视频进行跟踪
- 自动检索对应的 side 视频（PLAYER2 和 PLAYER3）
- 自动使用 side 模型对 side 视频进行跟踪

命名规则：
- top 视频包含 PLAYER1
- side 视频包含 PLAYER2 和 PLAYER3（将 PLAYER1 替换为 PLAYER2 或 PLAYER3）
- 其他部分（时间戳、行为类型等）相同

用法示例：
    python scripts/sleap/batch_track.py ^
      --top-videos "D:\behavior_clips\20251225_K_1_2_FCMC_clips\20251225_K_1_2_FCMC_PLAYER1_f_Spawn_816.572-820.572.mp4" ^
      --top-videos "D:\behavior_clips\20251225_K_1_2_FCMC_clips\20251225_K_1_2_FCMC_PLAYER1_m1_Sneak_804.705-812.873.mp4" ^
      --top-models "D:\sleap_top\models\centroid.model" ^
      --top-models "D:\sleap_top\models\instance.model" ^
      --side-models "D:\sleap_side\models\centroid.model" ^
      --side-models "D:\sleap_side\models\instance.model" ^
      --output-dir "D:\sleap_output" ^
      --max-instances 3
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

# SLEAP 进度行格式: "Predicting... ----  75% 696/929 ETA: ..."
_PROGRESS_RE = re.compile(r"(\d+)%\s+(\d+)/(\d+)\b")


def classify_video_view(video_path: Path) -> str:
    """
    根据视频路径判断使用 top 还是 side 模型。
    - 路径中含 top 或 cam0 → "top"
    - 路径中含 side、cam1、cam2 → "side"
    - 否则默认 "top"
    """
    s = str(video_path).lower()
    if "top" in s or "cam0" in s:
        return "top"
    if "side" in s or "cam1" in s or "cam2" in s:
        return "side"
    return "top"


def get_model_subdir(model_paths: list[Path]) -> str:
    """
    从模型路径推导输出子目录名，如 top_mouth_375、side_mouth_437。
    规则：模型目录名去掉 sleap_ 前缀 + 文件名中的 n=数字。
    例如 D:/sleap_top_mouth/models/xxx.centroid.n=375 → top_mouth_375
    """
    if not model_paths:
        return "unknown"
    p = Path(model_paths[0]).resolve()
    # 例如 .../sleap_top_mouth/models/260210_165744.centroid.n=375
    model_dir_name = p.parent.parent.name  # sleap_top_mouth
    base = model_dir_name.replace("sleap_", "").strip("_")  # top_mouth
    match = re.search(r"n[=_](\d+)", p.name, re.IGNORECASE)
    suffix = match.group(1) if match else ""
    return f"{base}_{suffix}" if suffix else base


def get_clips_subdir(video_path: Path) -> str:
    """
    从视频父目录名得到 clips 子目录名，如 20251225_K_1_2_FCMC_clips → 20251225_K_1_2_clips。
    """
    name = video_path.parent.name
    return name.replace("FCMC_", "")


def generate_prediction_slp_path(
    video_path: Path,
    output_base: Path,
    model_subdir: str,
) -> Path:
    """
    生成符合约定的 .prediction.slp 路径：
    output_base / model_subdir / clips_subdir / {model_subdir}_{video.stem}.prediction.slp
    例如 D:\\sleap_output\\top_mouth_375\\20251225_K_1_2_clips\\top_mouth_375_xxx.prediction.slp
    """
    clips_subdir = get_clips_subdir(video_path)
    out_dir = output_base / model_subdir / clips_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{model_subdir}_{video_path.stem}.prediction.slp"


def _sleap_cmd(exe: str, args: list[str]) -> list[str]:
    """根据当前环境构建 SLEAP 命令：若 PATH 中有可执行文件则直接调用，否则返回 [exe, *args] 由系统报错。"""
    if shutil.which(exe) is not None:
        return [exe, *args]
    return [exe, *args]


def find_side_videos(top_video_path: Path) -> tuple[Path | None, Path | None]:
    """
    根据 top 视频路径查找对应的 side 视频（PLAYER2 和 PLAYER3）。
    
    命名规则（依次尝试多种约定）：
    1. PLAYER 命名：
       - top 视频：..._PLAYER1_...
       - side 视频：..._PLAYER2_... 和 ..._PLAYER3_...（将 PLAYER1 替换为 PLAYER2 或 PLAYER3）
    2. cam0/cam1/cam2 + 视角后缀命名（无 PLAYER）：
       - top  视频路径示例：...\\top_cam\\cam0_top.mp4
       - side 视频路径示例：...\\side1_cam\\cam1_side1.mp4、...\\side2_cam\\cam2_side2.mp4
       - 规则：
         * 优先尝试把父目录中的 `top_cam` 替换为 `side1_cam` / `side2_cam`
         * 同时把文件名中的 `cam0` → `cam1` / `cam2`，`top` → `side1` / `side2`
    
    返回：
    - (side1_path, side2_path) 如果找到，否则为 (None, None)
    """
    top_str = str(top_video_path)
    
    # -------------------------------
    # 1) PLAYER1 / PLAYER2 / PLAYER3
    # -------------------------------
    if "PLAYER1" in top_str:
        side1_str = top_str.replace("PLAYER1", "PLAYER2")
        side2_str = top_str.replace("PLAYER1", "PLAYER3")
        side1_path = Path(side1_str)
        side2_path = Path(side2_str)
        side1_exists = side1_path.exists()
        side2_exists = side2_path.exists()
        if side1_exists or side2_exists:
            return (
                side1_path if side1_exists else None,
                side2_path if side2_exists else None,
            )

    # --------------------------------------------------------
    # 2) cam0 / cam1 / cam2 + top / side1 / side2 + *_cam 目录
    # --------------------------------------------------------
    top_path = Path(top_video_path)
    parent = top_path.parent
    stem = top_path.stem
    name = top_path.name

    candidates: list[tuple[Path, str]] = []

    # 尝试基于目录名的替换：top_cam -> side1_cam / side2_cam
    if parent.name.lower() == "top_cam":
        side1_dir = parent.with_name("side1_cam")
        side2_dir = parent.with_name("side2_cam")

        # 文件名 cam0_top -> cam1_side1 / cam2_side2
        side1_name = name.replace("cam0", "cam1").replace("top", "side1")
        side2_name = name.replace("cam0", "cam2").replace("top", "side2")

        candidates.append((side1_dir / side1_name, "cam_rule_side1"))
        candidates.append((side2_dir / side2_name, "cam_rule_side2"))

    # 兜底：只替换文件名中的 cam0/cam1/cam2 和 top/side1/side2，不动目录
    side1_name_fallback = name.replace("cam0", "cam1").replace("top", "side1")
    side2_name_fallback = name.replace("cam0", "cam2").replace("top", "side2")
    candidates.append((parent / side1_name_fallback, "name_only_side1"))
    candidates.append((parent / side2_name_fallback, "name_only_side2"))

    side1_path_final: Path | None = None
    side2_path_final: Path | None = None

    for cand_path, kind in candidates:
        if not cand_path.exists():
            continue
        if "side1" in cand_path.stem and side1_path_final is None:
            side1_path_final = cand_path
        if "side2" in cand_path.stem and side2_path_final is None:
            side2_path_final = cand_path

    return side1_path_final, side2_path_final


def run_sleap_tracking(
    video_path: Path,
    model_paths: list[Path],
    output_path: Path,
    max_instances: int = 3,
    frames: str | None = None,
    device: str = "gpu",
    progress_callback: Callable[[int, int], None] | None = None,
) -> None:
    """
    运行 sleap track 命令进行视频跟踪。
    
    参数：
    - video_path: 输入视频路径
    - model_paths: 模型路径列表（通常是 [centroid_model, instance_model]）
    - output_path: 输出 .slp 文件路径
    - max_instances: 最大实例数
    - frames: 可选。None 时不传 --frames，推理整段视频（与训练时的验证采样无关）；非 None 时传 --frames，格式依 sleap-nn 版本可能为 "0-999" 或 "0,1,2,3"，若报非法参数请保持 None 或查 sleap-nn track --help
    - device: 计算设备（"cpu", "gpu", "first-gpu", "last-gpu"），默认 "gpu"
    - progress_callback: 可选，(current, total) 在解析到进度行时调用，用于在 Notebook 中更新单一进度条，不刷屏
    """
    if not video_path.exists():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    
    for model_path in model_paths:
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 构建 sleap track 命令（新 CLI；旧版 sleap-nn-track 已弃用）
    cmd_args = [
        "-i", str(video_path),
        "-o", str(output_path),
        "-n", str(max_instances),
    ]
    
    # 添加模型参数（每个模型一个 -m）
    for model_path in model_paths:
        cmd_args.extend(["-m", str(model_path)])
    
    # 帧范围：仅当 frames 非 None 时才传 --frames；不传时 sleap-nn 按整段视频推理（与训练时「随机 20 帧」无关）
    if frames:
        cmd_args.extend(["--frames", frames])
    # 添加设备参数（GPU 支持）
    if device:
        cmd_args.extend(["--device", device])
    
    # 优先使用新 CLI「sleap track」；若无则退回到 sleap-nn-track
    if shutil.which("sleap") is not None:
        cmd = _sleap_cmd("sleap", ["track"] + cmd_args)
    else:
        cmd = _sleap_cmd("sleap-nn-track", cmd_args)
    
    print("\n执行跟踪命令:")
    print(f"  {' '.join(cmd)}")
    if not frames:
        print("  跟踪范围: 整段视频（未指定 --frames）")
    print()  # 空行，让进度条显示更清晰
    
    # 设置环境变量，帮助解决 GPU 加载问题；无缓冲输出便于在 Notebook 中看到进度条
    env = os.environ.copy()
    if device and device != "cpu":
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")
    env.setdefault("PYTHONUNBUFFERED", "1")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            bufsize=1,
        )
        if progress_callback is not None:
            # 解析进度行并只回调，不打印进度行，避免刷屏；其余行照常打印
            for line in proc.stdout:
                m = _PROGRESS_RE.search(line)
                if m:
                    _pct, current, total = m.groups()
                    progress_callback(int(current), int(total))
                else:
                    sys.stdout.write(line)
                    sys.stdout.flush()
        else:
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
        returncode = proc.wait()
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, cmd)
        print(f"\n✓ 成功生成: {output_path}")
    except subprocess.CalledProcessError as e:
        # 检查是否是模型数量问题
        error_msg = f"\n跟踪失败（返回码: {e.returncode}）"
        
        if len(model_paths) == 1:
            error_msg += (
                f"\n\n提示: 对于 top-down 跟踪流程，通常需要两个模型："
                f"\n  1. Centroid 模型（用于检测个体位置）"
                f"\n  2. Instance 模型（用于预测关键点）"
                f"\n当前只提供了一个模型: {model_paths[0].name}"
            )
        
        # 添加 GPU 相关错误的提示
        if device and device != "cpu":
            error_msg += (
                f"\n\n如果遇到 GPU 相关错误（如 'don't know how to restore data location'），"
                f"\n这是 PyTorch 模型加载时的设备位置恢复问题。可以尝试："
                f"\n  1. 升级 PyTorch: pip install --upgrade torch"
                f"\n  2. 使用 --device first-gpu 代替 --device gpu"
                f"\n  3. 使用 --device cpu（虽然慢，但可以工作）"
                f"\n  4. 检查 CUDA 版本兼容性: python -c \"import torch; print(torch.version.cuda)\""
                f"\n  5. 如果模型是在不同 GPU 上训练的，可能需要重新保存模型"
            )
        
        raise RuntimeError(error_msg) from e


def generate_output_path(
    video_path: Path,
    output_dir: Path,
    view_type: str = "top",
) -> Path:
    """
    根据视频路径和输出目录生成输出 .slp 文件路径。

    目前项目的视频路径通常形如：
        D:/videos_raw/20251220_E_1_2_FCMC/top_cam/cam0_top.mp4

    期望的输出命名示例：
        20251220_E_1_2_FCMC_top_cam.slp

    即：使用「实验文件夹名 + 视角文件夹名」，而不是 cam0_top 这样的文件名。

    参数：
    - video_path: 输入视频路径
    - output_dir: 输出目录
    - view_type: 视角类型（"top", "side1", "side2"）
    """
    parent = video_path.parent
    grandparent = parent.parent if parent is not None else None

    if grandparent is not None and parent is not None:
        # 例如：grandparent.name = "20251220_E_1_2_FCMC", parent.name = "top_cam"
        session_name = grandparent.name
        view_folder = parent.name
        output_filename = f"{session_name}_{view_folder}.slp"
    else:
        # 兜底：退回到旧的「文件名 + 视角」命名方式
        video_stem = video_path.stem
        output_filename = f"{video_stem}_{view_type}.slp"

    return output_dir / output_filename


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "批量 SLEAP 跟踪脚本\n"
            "自动对 top 视频进行跟踪，并查找对应的 side 视频进行跟踪。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--top-videos",
        action="append",
        required=True,
        help="Top 视角视频路径（可以指定多个，每个视频用 --top-videos 指定一次）",
    )
    parser.add_argument(
        "--top-models",
        action="append",
        required=True,
        help="Top 模型路径（通常是 centroid 和 instance 两个模型，每个模型用 --top-models 指定一次）",
    )
    parser.add_argument(
        "--side-models",
        action="append",
        required=True,
        help="Side 模型路径（通常是 centroid 和 instance 两个模型，每个模型用 --side-models 指定一次）",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="输出目录，所有 .slp 文件将保存在此目录",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=3,
        help="最大实例数（默认：3）",
    )
    parser.add_argument(
        "--frames",
        default=None,
        help="可选的帧范围，格式如 '19920-29919'",
    )
    parser.add_argument(
        "--skip-missing-side",
        action="store_true",
        help="如果 side 视频不存在，跳过而不报错",
    )
    parser.add_argument(
        "--device",
        default="gpu",
        choices=["cpu", "gpu", "first-gpu", "last-gpu"],
        help="计算设备（默认：gpu）。可选值：cpu, gpu, first-gpu, last-gpu",
    )
    
    args = parser.parse_args()
    
    # 转换路径
    top_videos = [Path(v).resolve() for v in args.top_videos]
    top_models = [Path(m).resolve() for m in args.top_models]
    side_models = [Path(m).resolve() for m in args.side_models]
    output_dir = Path(args.output_dir).resolve()
    
    # 验证输入
    for top_video in top_videos:
        if not top_video.exists():
            raise FileNotFoundError(f"Top 视频不存在: {top_video}")
    
    for model in top_models + side_models:
        if not model.exists():
            raise FileNotFoundError(f"模型文件不存在: {model}")
    
    # 检查模型数量并给出提示
    if len(top_models) == 1:
        print(f"\n⚠️  警告: Top 模型只提供了一个 ({top_models[0].name})")
        print(f"  对于 top-down 跟踪流程，通常需要两个模型：")
        print(f"    1. Centroid 模型（用于检测个体位置）")
        print(f"    2. Instance 模型（用于预测关键点）")
        print(f"  如果只有一个模型，请确保它是 single instance 模型。\n")
    
    if len(side_models) == 1:
        print(f"\n⚠️  警告: Side 模型只提供了一个 ({side_models[0].name})")
        print(f"  对于 top-down 跟踪流程，通常需要两个模型：")
        print(f"    1. Centroid 模型（用于检测个体位置）")
        print(f"    2. Instance 模型（用于预测关键点）")
        print(f"  如果只有一个模型，请确保它是 single instance 模型。\n")
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"开始批量跟踪处理...")
    print(f"  Top 视频数量: {len(top_videos)}")
    print(f"  Top 模型数量: {len(top_models)}")
    print(f"  Side 模型数量: {len(side_models)}")
    print(f"  输出目录: {output_dir}")
    print(f"  最大实例数: {args.max_instances}")
    
    # 处理每个 top 视频
    for i, top_video in enumerate(top_videos, 1):
        print(f"\n{'='*60}")
        print(f"处理 Top 视频 {i}/{len(top_videos)}: {top_video.name}")
        print(f"{'='*60}")
        
        # 1. 对 top 视频进行跟踪
        top_output = generate_output_path(top_video, output_dir, "top")
        print(f"\n[1/3] 跟踪 Top 视频...")
        run_sleap_tracking(
            video_path=top_video,
            model_paths=top_models,
            output_path=top_output,
            max_instances=args.max_instances,
            frames=args.frames,
            device=args.device,
        )
        
        # 2. 查找对应的 side 视频
        side1_video, side2_video = find_side_videos(top_video)
        
        # 3. 对 side1 视频进行跟踪（如果存在）
        if side1_video is not None:
            print(f"\n[2/3] 跟踪 Side1 视频 (PLAYER2)...")
            side1_output = generate_output_path(side1_video, output_dir, "side1")
            run_sleap_tracking(
                video_path=side1_video,
                model_paths=side_models,
                output_path=side1_output,
                max_instances=args.max_instances,
                frames=args.frames,
                device=args.device,
            )
        else:
            if args.skip_missing_side:
                print(f"\n[2/3] 跳过 Side1 视频（未找到）")
            else:
                raise FileNotFoundError(
                    f"未找到对应的 Side1 视频 (PLAYER2): {top_video}\n"
                    f"预期路径: {str(top_video).replace('PLAYER1', 'PLAYER2')}"
                )
        
        # 4. 对 side2 视频进行跟踪（如果存在）
        if side2_video is not None:
            print(f"\n[3/3] 跟踪 Side2 视频 (PLAYER3)...")
            side2_output = generate_output_path(side2_video, output_dir, "side2")
            run_sleap_tracking(
                video_path=side2_video,
                model_paths=side_models,
                output_path=side2_output,
                max_instances=args.max_instances,
                frames=args.frames,
                device=args.device,
            )
        else:
            if args.skip_missing_side:
                print(f"\n[3/3] 跳过 Side2 视频（未找到）")
            else:
                raise FileNotFoundError(
                    f"未找到对应的 Side2 视频 (PLAYER3): {top_video}\n"
                    f"预期路径: {str(top_video).replace('PLAYER1', 'PLAYER3')}"
                )
    
    print(f"\n{'='*60}")
    print(f"✓ 所有视频处理完成！")
    print(f"  输出目录: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

