from __future__ import annotations

"""
Batch SLEAP tracking.

This helper:

- accepts one or more top-view videos
- runs the top-view model(s) on PLAYER1 clips
- locates the matching side-view videos for PLAYER2 and PLAYER3
- runs the side-view model(s) on those matching clips

Typical usage:
    python scripts/sleap/batch_track.py ^
      --top-videos "D:\behavior_clips\...\PLAYER1_....mp4" ^
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

_PROGRESS_RE = re.compile(r"(\d+)%\s+(\d+)/(\d+)\b")


def classify_video_view(video_path: Path) -> str:
    """
    Infer whether a video should use top or side models.

    - paths containing `top` or `cam0` -> `top`
    - paths containing `side`, `cam1`, or `cam2` -> `side`
    - default fallback -> `top`
    """
    s = str(video_path).lower()
    if "top" in s or "cam0" in s:
        return "top"
    if "side" in s or "cam1" in s or "cam2" in s:
        return "side"
    return "top"


def get_model_subdir(model_paths: list[Path]) -> str:
    """Infer an output subdirectory name such as `top_mouth_375` or `side_mouth_437`."""
    if not model_paths:
        return "unknown"
    p = Path(model_paths[0]).resolve()
    model_dir_name = p.parent.parent.name
    base = model_dir_name.replace("sleap_", "").strip("_")
    match = re.search(r"n[=_](\d+)", p.name, re.IGNORECASE)
    suffix = match.group(1) if match else ""
    return f"{base}_{suffix}" if suffix else base


def get_clips_subdir(video_path: Path) -> str:
    """Infer the clips subdirectory name from the video parent directory."""
    return video_path.parent.name.replace("FCMC_", "")


def generate_prediction_slp_path(
    video_path: Path,
    output_base: Path,
    model_subdir: str,
) -> Path:
    """
    Build the conventional `.prediction.slp` output path:
    `output_base / model_subdir / clips_subdir / {model_subdir}_{video.stem}.prediction.slp`
    """
    clips_subdir = get_clips_subdir(video_path)
    out_dir = output_base / model_subdir / clips_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{model_subdir}_{video_path.stem}.prediction.slp"


def _sleap_cmd(exe: str, args: list[str]) -> list[str]:
    """Use the executable from PATH if present; otherwise return `[exe, *args]`."""
    if shutil.which(exe) is not None:
        return [exe, *args]
    return [exe, *args]


def find_side_videos(top_video_path: Path) -> tuple[Path | None, Path | None]:
    """
    Find the matching side videos (PLAYER2 and PLAYER3) for a top-view clip.

    Supported naming patterns include:
    1. `PLAYER1` -> `PLAYER2` / `PLAYER3`
    2. `top_cam/cam0_top` -> `side1_cam/cam1_side1` and `side2_cam/cam2_side2`
    """
    top_str = str(top_video_path)

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

    top_path = Path(top_video_path)
    parent = top_path.parent
    name = top_path.name

    side1_candidates: list[Path] = []
    side2_candidates: list[Path] = []

    if parent.name.lower() == "top_cam":
        side1_dir = parent.with_name("side1_cam")
        side2_dir = parent.with_name("side2_cam")

        # medaka recording convention: top_cam/cam0_top.mp4 paired with
        # side1_cam/cam1_side.mp4 and side2_cam/cam2_side.mp4.
        side1_candidates.append(side1_dir / name.replace("cam0_top", "cam1_side"))
        side2_candidates.append(side2_dir / name.replace("cam0_top", "cam2_side"))

        # legacy squid-style naming: cam1_side1 / cam2_side2.
        side1_candidates.append(
            side1_dir / name.replace("cam0", "cam1").replace("top", "side1")
        )
        side2_candidates.append(
            side2_dir / name.replace("cam0", "cam2").replace("top", "side2")
        )

    side1_candidates.append(parent / name.replace("cam0_top", "cam1_side"))
    side2_candidates.append(parent / name.replace("cam0_top", "cam2_side"))
    side1_candidates.append(
        parent / name.replace("cam0", "cam1").replace("top", "side1")
    )
    side2_candidates.append(
        parent / name.replace("cam0", "cam2").replace("top", "side2")
    )

    def _first_existing(paths: list[Path]) -> Path | None:
        for p in paths:
            if p.exists():
                return p
        return None

    return _first_existing(side1_candidates), _first_existing(side2_candidates)


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
    Run `sleap track` on one video.

    If `frames` is None, the full clip is tracked.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file does not exist: {video_path}")

    for model_path in model_paths:
        if not model_path.exists():
            raise FileNotFoundError(f"Model file does not exist: {model_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd_args = [
        "-i", str(video_path),
        "-o", str(output_path),
        "-n", str(max_instances),
    ]

    for model_path in model_paths:
        cmd_args.extend(["-m", str(model_path)])

    if frames:
        cmd_args.extend(["--frames", frames])
    if device:
        cmd_args.extend(["--device", device])

    if shutil.which("sleap") is not None:
        cmd = _sleap_cmd("sleap", ["track"] + cmd_args)
    else:
        cmd = _sleap_cmd("sleap-nn-track", cmd_args)

    print("\nRunning tracking command:")
    print(f"  {' '.join(cmd)}")
    if not frames:
        print("  Frame range: full clip (no --frames argument)")
    print()

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
        print(f"\n✓ Created: {output_path}")
    except subprocess.CalledProcessError as e:
        error_msg = f"\nTracking failed (exit code: {e.returncode})"

        if len(model_paths) == 1:
            error_msg += (
                f"\n\nFor top-down tracking, you usually need two models:"
                f"\n  1. a centroid model"
                f"\n  2. an instance model"
                f"\nOnly one model was provided: {model_paths[0].name}"
            )

        if device and device != "cpu":
            error_msg += (
                f"\n\nIf you encounter GPU-related errors such as "
                f"'don't know how to restore data location', try:"
                f"\n  1. upgrading PyTorch: pip install --upgrade torch"
                f"\n  2. using --device first-gpu instead of --device gpu"
                f"\n  3. using --device cpu as a fallback"
                f"\n  4. checking CUDA compatibility: python -c \"import torch; print(torch.version.cuda)\""
                f"\n  5. re-saving the model if it was trained on a different GPU setup"
            )

        raise RuntimeError(error_msg) from e


def generate_output_path(
    video_path: Path,
    output_dir: Path,
    view_type: str = "top",
) -> Path:
    """
    Generate the output `.slp` path from the input video path.

    Preferred naming uses:
        `{session_name}_{view_folder}.slp`
    """
    parent = video_path.parent
    grandparent = parent.parent if parent is not None else None

    if grandparent is not None and parent is not None:
        session_name = grandparent.name
        view_folder = parent.name
        output_filename = f"{session_name}_{view_folder}.slp"
    else:
        video_stem = video_path.stem
        output_filename = f"{video_stem}_{view_type}.slp"

    return output_dir / output_filename


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Batch SLEAP tracking.\n"
            "Track top-view videos and automatically locate the matching side-view clips."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--top-videos",
        action="append",
        required=True,
        help="Top-view video path. Repeat this flag to provide multiple videos.",
    )
    parser.add_argument(
        "--top-models",
        action="append",
        required=True,
        help="Top-view model path. Repeat as needed for centroid/instance models.",
    )
    parser.add_argument(
        "--side-models",
        action="append",
        required=True,
        help="Side-view model path. Repeat as needed for centroid/instance models.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for all generated .slp files.",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=3,
        help="Maximum number of instances to track (default: 3).",
    )
    parser.add_argument(
        "--frames",
        default=None,
        help="Optional frame range, for example '19920-29919'.",
    )
    parser.add_argument(
        "--skip-missing-side",
        action="store_true",
        help="Skip clips whose side-view videos are missing.",
    )
    parser.add_argument(
        "--device",
        default="gpu",
        choices=["cpu", "gpu", "first-gpu", "last-gpu"],
        help="Compute device to use (default: gpu).",
    )

    args = parser.parse_args()

    top_videos = [Path(v).resolve() for v in args.top_videos]
    top_models = [Path(m).resolve() for m in args.top_models]
    side_models = [Path(m).resolve() for m in args.side_models]
    output_dir = Path(args.output_dir).resolve()

    for top_video in top_videos:
        if not top_video.exists():
            raise FileNotFoundError(f"Top-view video does not exist: {top_video}")

    for model in top_models + side_models:
        if not model.exists():
            raise FileNotFoundError(f"Model file does not exist: {model}")

    if len(top_models) == 1:
        print(f"\nWarning: only one top model was provided ({top_models[0].name}).")
        print("  Top-down tracking usually needs:")
        print("    1. a centroid model")
        print("    2. an instance model")
        print("  If that is intentional, make sure it is a single-instance model.\n")

    if len(side_models) == 1:
        print(f"\nWarning: only one side model was provided ({side_models[0].name}).")
        print("  Top-down tracking usually needs:")
        print("    1. a centroid model")
        print("    2. an instance model")
        print("  If that is intentional, make sure it is a single-instance model.\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Starting batch tracking...")
    print(f"  Number of top videos: {len(top_videos)}")
    print(f"  Number of top models: {len(top_models)}")
    print(f"  Number of side models: {len(side_models)}")
    print(f"  Output directory: {output_dir}")
    print(f"  Max instances: {args.max_instances}")

    for i, top_video in enumerate(top_videos, 1):
        print(f"\n{'='*60}")
        print(f"Processing top video {i}/{len(top_videos)}: {top_video.name}")
        print(f"{'='*60}")

        top_output = generate_output_path(top_video, output_dir, "top")
        print("\n[1/3] Tracking top-view clip...")
        run_sleap_tracking(
            video_path=top_video,
            model_paths=top_models,
            output_path=top_output,
            max_instances=args.max_instances,
            frames=args.frames,
            device=args.device,
        )

        side1_video, side2_video = find_side_videos(top_video)

        if side1_video is not None:
            print("\n[2/3] Tracking side1 clip (PLAYER2)...")
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
                print("\n[2/3] Skipping side1 clip (not found)")
            else:
                raise FileNotFoundError(
                    f"Matching Side1 video (PLAYER2) not found for: {top_video}\n"
                    f"Expected path: {str(top_video).replace('PLAYER1', 'PLAYER2')}"
                )

        if side2_video is not None:
            print("\n[3/3] Tracking side2 clip (PLAYER3)...")
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
                print("\n[3/3] Skipping side2 clip (not found)")
            else:
                raise FileNotFoundError(
                    f"Matching Side2 video (PLAYER3) not found for: {top_video}\n"
                    f"Expected path: {str(top_video).replace('PLAYER1', 'PLAYER3')}"
                )

    print(f"\n{'='*60}")
    print("✓ All videos processed successfully.")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
