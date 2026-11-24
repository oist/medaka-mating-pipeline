"""Build low-resolution review proxies of the three raw camera videos.

For every view one ffmpeg pass decodes the source once and writes two files:

* ``<view>.play.mp4`` - every ``step``-th source frame (``step`` =
  round(source_fps / playback_fps), 6 at 166 fps -> 27.7 fps). Proxy frame
  ``k`` is exactly source frame ``k * step`` (``select`` by frame number, no
  timestamp-based resampling).
* ``<view>.full.mp4`` - every source frame, for frame-accurate stepping.

Both use H.264 with short GOPs and no B-frames so any laptop can decode and
seek them. Files are written as ``*.partial.mp4`` and renamed after their
frame counts are verified; ``proxy_manifest.json`` is written last and marks
the experiment as done, so an interrupted run simply restarts that experiment.

ffmpeg runs at idle CPU priority, one view at a time. It still shares the GPU:
a full-length transcode slowed a concurrent local SLEAP inference from 32.5 to
~29 frames/s (~11%), so ``build_packages`` waits while local inference runs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

VIEWS = ("top_cam", "side1_cam", "side2_cam")
VIDEO_NAMES = {"top_cam": "cam0_top.mp4", "side1_cam": "cam1_side.mp4", "side2_cam": "cam2_side.mp4"}
MANIFEST_NAME = "proxy_manifest.json"
PROXY_VERSION = 1


@dataclass(frozen=True)
class SourceInfo:
    path: Path
    fps: float
    total_frames: int
    width: int
    height: int


def find_ffmpeg() -> str:
    """ffmpeg on PATH, else the binary bundled with imageio-ffmpeg."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError("ffmpeg not found on PATH and imageio-ffmpeg is not installed") from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def has_nvenc(ffmpeg: str) -> bool:
    try:
        out = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return False
    return "h264_nvenc" in out and shutil.which("nvidia-smi") is not None


def playback_step(fps: float, playback_fps: float) -> int:
    return max(1, round(fps / playback_fps))


def proxy_size(width: int, height: int, target: int) -> tuple[int, int]:
    """Scale the longer side to ``target`` (even numbers, aspect kept)."""
    scale = target / max(width, height)
    return max(2, round(width * scale / 2) * 2), max(2, round(height * scale / 2) * 2)


def source_info(exp_dir: Path, view: str) -> SourceInfo:
    """Video facts from ``experiment_metadata.json``; falls back to probing the file."""
    video = exp_dir / view / VIDEO_NAMES[view]
    meta_path = exp_dir / "experiment_metadata.json"
    if meta_path.exists():
        try:
            v = json.loads(meta_path.read_text(encoding="utf-8"))["views"][view]
            return SourceInfo(video, float(v["frame_rate"]), int(v["total_frames"]),
                              int(v["frame_width"]), int(v["frame_height"]))
        except (KeyError, TypeError, ValueError):
            pass
    import cv2

    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open {video}")
        return SourceInfo(video, float(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                          int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        cap.release()


def build_ffmpeg_cmd(
    ffmpeg: str,
    src: Path,
    full_out: Path,
    play_out: Path,
    *,
    width: int,
    height: int,
    step: int,
    source_fps: float,
    gpu: bool,
    limit_seconds: float | None = None,
) -> list[str]:
    play_rate = Fraction(source_fps).limit_denominator(1000) / step
    select = f"select='not(mod(n\\,{step}))',setpts=N/(({play_rate.numerator}/{play_rate.denominator})*TB)"
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    if gpu:
        cmd += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        scale = f"scale_cuda={width}:{height}"
        enc = ["-c:v", "h264_nvenc", "-preset", "p4", "-bf", "0"]
        full_q, play_q = ["-cq", "28"], ["-cq", "24"]
    else:
        scale = f"scale={width}:{height}"
        enc = ["-c:v", "libx264", "-preset", "veryfast", "-bf", "0", "-pix_fmt", "yuv420p"]
        full_q, play_q = ["-crf", "28"], ["-crf", "24"]
    if limit_seconds:
        cmd += ["-t", f"{limit_seconds:g}"]
    cmd += ["-i", str(src), "-filter_complex", f"[0:v]{scale},split=2[full][p];[p]{select}[play]"]
    common = ["-fps_mode", "passthrough", "-an", "-movflags", "+faststart"]
    cmd += ["-map", "[full]", *common, *enc, *full_q, "-g", "30", str(full_out)]
    cmd += ["-map", "[play]", *common, *enc, *play_q, "-g", str(max(1, round(float(play_rate)))), str(play_out)]
    return cmd


def low_priority_kwargs() -> dict:
    if os.name == "nt":
        return {"creationflags": subprocess.IDLE_PRIORITY_CLASS}
    return {"preexec_fn": lambda: os.nice(19)}


def frame_count(path: Path) -> int:
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else -1
    finally:
        cap.release()


def build_view(ffmpeg: str, info: SourceInfo, out_dir: Path, view: str, *, size: int, playback_fps: float,
               gpu: bool, limit_seconds: float | None, log) -> dict:
    step = playback_step(info.fps, playback_fps)
    width, height = proxy_size(info.width, info.height, size)
    full_final, play_final = out_dir / f"{view}.full.mp4", out_dir / f"{view}.play.mp4"
    full_tmp, play_tmp = out_dir / f"{view}.full.partial.mp4", out_dir / f"{view}.play.partial.mp4"
    expected_full = info.total_frames if not limit_seconds else None

    last_err = ""
    for use_gpu in ([True, False] if gpu else [False]):
        cmd = build_ffmpeg_cmd(ffmpeg, info.path, full_tmp, play_tmp, width=width, height=height, step=step,
                               source_fps=info.fps, gpu=use_gpu, limit_seconds=limit_seconds)
        log(f"  {view}: ffmpeg ({'nvenc' if use_gpu else 'cpu'}) {width}x{height}, play step {step}")
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace", **low_priority_kwargs())
        if proc.returncode == 0:
            break
        last_err = proc.stderr.strip()[-2000:]
        log(f"  {view}: ffmpeg failed (rc {proc.returncode}): {last_err}")
    else:
        raise RuntimeError(f"{view}: ffmpeg failed: {last_err}")

    n_full, n_play = frame_count(full_tmp), frame_count(play_tmp)
    if expected_full is not None and n_full != expected_full:
        # A truncated encode is an error; an off-by-a-few metadata count is only recorded.
        if n_full < 0.99 * expected_full:
            raise RuntimeError(f"{view}: full proxy has {n_full} frames, source metadata says {expected_full}")
        log(f"  {view}: WARNING full proxy has {n_full} frames, metadata says {expected_full}")
    if n_play != -(-n_full // step):
        raise RuntimeError(f"{view}: play proxy has {n_play} frames, expected ceil({n_full}/{step})")
    os.replace(full_tmp, full_final)
    os.replace(play_tmp, play_final)
    elapsed = time.time() - t0
    log(f"  {view}: done in {elapsed / 60:.1f} min ({n_full} frames, "
        f"{(full_final.stat().st_size + play_final.stat().st_size) / 1e9:.2f} GB)")
    return {
        "source_video": str(info.path), "source_fps": info.fps, "source_frames": n_full,
        "metadata_frames": info.total_frames,
        "source_width": info.width, "source_height": info.height,
        "proxy_width": width, "proxy_height": height, "play_step": step,
        "play_frames": n_play, "full_file": full_final.name, "play_file": play_final.name,
        "encoder": "h264_nvenc" if use_gpu else "libx264", "seconds": round(elapsed, 1),
    }


def build_experiment(exp: str, data_dir: Path, review_root: Path, *, size: int = 800, playback_fps: float = 30.0,
                     gpu: bool | None = None, limit_seconds: float | None = None, log=print,
                     before_view=None) -> str:
    """Build all three views' proxies. ``before_view()`` runs before each view (e.g. to wait for a free GPU)."""
    out_dir = review_root / exp / "proxy"
    manifest = out_dir / MANIFEST_NAME
    if manifest.exists():
        return "done"
    out_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = find_ffmpeg()
    gpu = has_nvenc(ffmpeg) if gpu is None else gpu
    views = {}
    for view in VIEWS:
        info = source_info(data_dir / exp, view)
        if not info.path.exists():
            return f"FAILED: missing {info.path}"
        if before_view is not None:
            before_view()
        views[view] = build_view(ffmpeg, info, out_dir, view, size=size, playback_fps=playback_fps, gpu=gpu,
                                 limit_seconds=limit_seconds, log=log)
    frames = {v["source_frames"] for v in views.values()}
    record = {
        "proxy_version": PROXY_VERSION, "experiment": exp, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "size": size, "playback_fps_target": playback_fps, "limit_seconds": limit_seconds,
        "frames_consistent_across_views": len(frames) == 1, "views": views,
    }
    tmp = manifest.with_suffix(".partial")
    tmp.write_text(json.dumps(record, indent=2), encoding="utf-8")
    os.replace(tmp, manifest)
    return "done (new)"


def main(argv: list[str] | None = None) -> int:
    from scripts.common.experiment_sets import resolve_experiments
    from scripts.common.local_paths import load_local_paths
    from scripts.sleap.batch_bottomup_track import default_paths_config

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiments", nargs="+", help="Folder names and/or set tokens (P12, K1, ALL).")
    p.add_argument("--paths-config", type=Path, default=None)
    p.add_argument("--review-root", type=Path, default=None, help="Default: <data_root>/review_packages.")
    p.add_argument("--size", type=int, default=800, help="Longer side of the proxies in pixels.")
    p.add_argument("--playback-fps", type=float, default=30.0)
    p.add_argument("--cpu", action="store_true", help="Force CPU decode/encode (no NVENC).")
    p.add_argument("--limit-seconds", type=float, default=None, help="Testing only: transcode the first N seconds.")
    p.add_argument("--include-part2", action="store_true")
    args = p.parse_args(argv)

    paths = load_local_paths(args.paths_config or default_paths_config())
    if paths is None:
        raise SystemExit("paths config not found")
    review_root = args.review_root or paths.data_root / "review_packages"
    exps = resolve_experiments(args.experiments, paths.medaka_data_dir, args.include_part2)
    failed = 0
    for exp in exps:
        print(f"[{time.strftime('%m-%d %H:%M')}] {exp}", flush=True)
        try:
            status = build_experiment(exp, paths.medaka_data_dir, review_root, size=args.size,
                                      playback_fps=args.playback_fps, gpu=False if args.cpu else None,
                                      limit_seconds=args.limit_seconds, log=lambda m: print(m, flush=True))
        except Exception as exc:  # keep going with the next experiment
            status = f"FAILED: {exc}"
        failed += status.startswith("FAILED")
        print(f"[{time.strftime('%m-%d %H:%M')}] {exp}: {status}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
