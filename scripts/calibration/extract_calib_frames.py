"""
Extract one frame from each calibration video for manual landmark annotation.
Supports batch processing across multiple input folders.
"""

import argparse
from pathlib import Path
from typing import List

import cv2


def extract_frames(input_dir: Path, output_dir: Path, frame_idx: int = 0) -> None:
    """
    Extract a specific frame from each camera video.

    Output filename format:
        {session}_cam{N}_{type}_frame{M}.png
    Example:
        cali_251217_way1_cam0_top_frame100.png
    """
    session_prefix = input_dir.name
    camera_dirs = {
        "top_cam": ("cam0", "top"),
        "side1_cam": ("cam1", "side"),
        "side2_cam": ("cam2", "side"),
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    for cam_dir, (cam_id, cam_type) in camera_dirs.items():
        cam_path = input_dir / cam_dir
        if not cam_path.exists():
            print(f"  [WARN] Camera folder not found: {cam_path}")
            continue

        video_files = list(cam_path.glob("*.mp4"))
        if not video_files:
            video_files = list(cam_path.glob("*.avi"))
        if not video_files:
            print(f"  [WARN] No video found in: {cam_path}")
            continue

        video_path = video_files[0]
        print(f"  Processing {video_path.name}...")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  [ERROR] Cannot open video: {video_path}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        actual_frame = frame_idx
        if frame_idx >= total_frames:
            print(f"  [WARN] Frame {frame_idx} is out of range (total: {total_frames}); using frame 0")
            actual_frame = 0

        cap.set(cv2.CAP_PROP_POS_FRAMES, actual_frame)
        ret, frame = cap.read()
        cap.release()

        if ret:
            output_path = output_dir / f"{session_prefix}_{cam_id}_{cam_type}_frame{actual_frame}.png"
            cv2.imwrite(str(output_path), frame)
            print(f"    Saved: {output_path}")
        else:
            print(f"  [ERROR] Failed to read frame {actual_frame} from {video_path}")


def parse_paths(path_str: str) -> List[str]:
    """Parse a comma-separated list of paths."""
    if not path_str:
        return []
    paths = [p.strip() for p in path_str.split(",")]
    return [p for p in paths if p]


def main():
    parser = argparse.ArgumentParser(
        description="Extract calibration frames (supports batch processing)"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Input folder path(s), comma-separated for multiple folders"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output folder path(s), comma-separated. Default: <input>_pic"
    )
    parser.add_argument(
        "--frame", type=int, default=0,
        help="Frame index to extract (default: 0)"
    )
    args = parser.parse_args()

    input_paths = parse_paths(args.input)
    if not input_paths:
        print("[ERROR] No input paths provided")
        return 1

    if args.output:
        output_paths = parse_paths(args.output)
        if len(output_paths) != len(input_paths):
            print(f"[ERROR] Number of output paths ({len(output_paths)}) must match input paths ({len(input_paths)})")
            return 1
    else:
        output_paths = [f"{p}_pic" for p in input_paths]

    print("=" * 70)
    print("Extract Calibration Frames")
    print("=" * 70)
    print(f"Frame index: {args.frame}")
    print(f"Processing {len(input_paths)} folder(s)...")
    print("=" * 70)

    for i, (input_path, output_path) in enumerate(zip(input_paths, output_paths)):
        input_dir = Path(input_path)
        output_dir = Path(output_path)

        print(f"\n[{i+1}/{len(input_paths)}] {input_dir.name}")
        print(f"  Input:  {input_dir}")
        print(f"  Output: {output_dir}")

        if not input_dir.exists():
            print(f"  [ERROR] Input folder not found: {input_dir}")
            continue

        extract_frames(input_dir, output_dir, args.frame)

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    exit(main())
