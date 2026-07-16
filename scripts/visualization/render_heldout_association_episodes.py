"""Render and concatenate discontiguous held-out association episodes.

This is a thin driver around :mod:`scripts.visualization.render_qc_4panel_fast`.
Each episode is rendered independently with its original source-frame range,
raw SLEAP instances in gray, and thin resolved-track markers.  Clips are then
concatenated per named method without altering any source track CSV.

Example::

    python -m scripts.visualization.render_heldout_association_episodes \
      20251225_K_1_2_FCMC \
      --episodes-csv runs/.../heldout_episodes.csv \
      --slp-dir Z:/.../round7_260704 --video-dir Z:/.../20251225_K_1_2_FCMC \
      --method baseline runs/.../tracks.csv id_final_pre_motion \
      --method optimized runs/.../tracks.csv id_final \
      --output-dir runs/.../heldout_episode_videos

Use ``--dry-run`` to validate episodes and inspect every renderer/ffmpeg
command without requiring videos, SLP files, track CSVs, or ffmpeg.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


DEFAULT_FFMPEG = (
    shutil.which("ffmpeg")
    or r"C:\ffmpeg\ffmpeg-8.0.1-full_build-shared\bin\ffmpeg.exe"
)


@dataclass(frozen=True)
class Episode:
    episode_id: str
    start_frame: int
    end_frame: int

    @property
    def n_frames(self) -> int:
        return self.end_frame - self.start_frame + 1


@dataclass(frozen=True)
class MethodSpec:
    name: str
    track_csv: Path
    id_column: str
    slug: str


@dataclass(frozen=True)
class RenderSettings:
    panel: int = 600
    fps: int = 60
    frame_step: int = 1
    tail_seconds: float = 5.0
    source_fps: float = 166.0
    gate_px: float = 80.0
    arrow_thickness: int = 1
    codec: str = "h264_nvenc"
    nvenc_cq: int = 24
    workers: int = 1

    def __post_init__(self) -> None:
        for name in ("panel", "fps", "frame_step", "arrow_thickness", "nvenc_cq", "workers"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be at least 1")
        for name in ("tail_seconds", "source_fps", "gate_px"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


def _integer_series(values: pd.Series, column: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not np.allclose(numeric, np.round(numeric)):
        raise ValueError(f"{column} must contain finite integer values")
    return numeric.astype(np.int64)


def load_episodes(path: Path) -> list[Episode]:
    """Load, sort, and strictly validate non-overlapping held-out episodes."""

    table = pd.read_csv(path, low_memory=False)
    required = {"episode_id", "start_frame", "end_frame"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError("episode CSV is missing columns: " + ", ".join(sorted(missing)))
    if table.empty:
        raise ValueError("episode CSV is empty")
    table = table.copy()
    table["episode_id"] = table["episode_id"].astype(str).str.strip()
    if table["episode_id"].eq("").any() or table["episode_id"].duplicated().any():
        raise ValueError("episode_id values must be non-empty and unique")
    table["start_frame"] = _integer_series(table["start_frame"], "start_frame")
    table["end_frame"] = _integer_series(table["end_frame"], "end_frame")
    if (table["start_frame"] < 0).any():
        raise ValueError("episode start_frame must be non-negative")
    if (table["end_frame"] < table["start_frame"]).any():
        raise ValueError("episode end_frame precedes start_frame")
    expected_n = table["end_frame"] - table["start_frame"] + 1
    if "n_frames" in table:
        declared = _integer_series(table["n_frames"], "n_frames")
        if not declared.equals(expected_n):
            raise ValueError("episode n_frames does not equal end_frame - start_frame + 1")
    table = table.sort_values(["start_frame", "end_frame", "episode_id"], kind="stable")
    starts = table["start_frame"].to_numpy(np.int64)
    ends = table["end_frame"].to_numpy(np.int64)
    if len(table) > 1 and np.any(starts[1:] <= ends[:-1]):
        raise ValueError("held-out episodes overlap; merge them before rendering")
    return [
        Episode(str(row.episode_id), int(row.start_frame), int(row.end_frame))
        for row in table.itertuples(index=False)
    ]


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    if not slug:
        raise ValueError("method name must contain a filename-safe character")
    return slug


def parse_methods(values: Sequence[Sequence[str]]) -> list[MethodSpec]:
    """Parse repeatable ``NAME TRACK_CSV ID_COLUMN`` method specifications."""

    if not values:
        raise ValueError("at least one --method is required")
    methods = [
        MethodSpec(name=str(name), track_csv=Path(track), id_column=str(column), slug=_slug(str(name)))
        for name, track, column in values
    ]
    names = [method.name for method in methods]
    slugs = [method.slug.casefold() for method in methods]
    if len(set(names)) != len(names):
        raise ValueError("method names must be unique")
    if len(set(slugs)) != len(slugs):
        raise ValueError("method names produce colliding output slugs")
    if any(not method.id_column.strip() for method in methods):
        raise ValueError("method identity columns must be non-empty")
    return methods


def build_render_command(
    *,
    python_executable: str,
    experiment: str,
    episode: Episode,
    method: MethodSpec,
    slp_dir: Path,
    video_dir: Path,
    output_path: Path,
    ffmpeg: str,
    settings: RenderSettings,
    calibration: Path | None = None,
    identity_specs: Sequence[str] = (),
    node_names: Sequence[str] = ("mouth", "tail"),
) -> list[str]:
    """Construct one exact source-range four-panel render command."""

    command = [
        python_executable,
        "-m",
        "scripts.visualization.render_qc_4panel_fast",
        experiment,
        "--slp-dir", str(slp_dir),
        "--track-csv", str(method.track_csv),
        "--id-column", method.id_column,
        "--video-dir", str(video_dir),
        "--out", str(output_path),
        "--panel", str(settings.panel),
        "--fps", str(settings.fps),
        "--frame-step", str(settings.frame_step),
        "--start-frame", str(episode.start_frame),
        "--end-frame", str(episode.end_frame),
        "--tail-seconds", str(settings.tail_seconds),
        "--source-fps", str(settings.source_fps),
        "--gate-px", str(settings.gate_px),
        "--ffmpeg", str(ffmpeg),
        "--codec", settings.codec,
        "--nvenc-cq", str(settings.nvenc_cq),
        "--workers", str(settings.workers),
        "--draw-raw-sleap",
        "--arrow-thickness", str(settings.arrow_thickness),
        "--node-names", *map(str, node_names),
    ]
    if calibration is not None:
        command.extend(("--calib", str(calibration)))
    for spec in identity_specs:
        command.extend(("--identity", str(spec)))
    return command


def build_concat_command(ffmpeg: str, concat_file: Path, output_path: Path) -> list[str]:
    return [
        str(ffmpeg), "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file),
        "-c", "copy", str(output_path),
    ]


def _build_reencode_command(
    ffmpeg: str, concat_file: Path, output_path: Path, codec: str
) -> list[str]:
    return [
        str(ffmpeg), "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file),
        "-c:v", codec, "-pix_fmt", "yuv420p", str(output_path),
    ]


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(path)


def _ffconcat_text(paths: Sequence[Path]) -> str:
    # The concat demuxer accepts POSIX separators on Windows.  Escape a rare
    # apostrophe using the quoting convention documented for ffconcat lists.
    lines = []
    for path in paths:
        value = path.resolve().as_posix().replace("'", "'\\''")
        lines.append(f"file '{value}'\n")
    return "".join(lines)


def _assert_runtime_inputs(
    methods: Sequence[MethodSpec], slp_dir: Path, video_dir: Path, ffmpeg: str
) -> None:
    if not slp_dir.is_dir():
        raise FileNotFoundError(f"SLP directory does not exist: {slp_dir}")
    if not video_dir.is_dir():
        raise FileNotFoundError(f"video directory does not exist: {video_dir}")
    for method in methods:
        if not method.track_csv.is_file():
            raise FileNotFoundError(f"method track CSV does not exist: {method.track_csv}")
        columns = pd.read_csv(method.track_csv, nrows=0).columns
        if method.id_column not in columns:
            raise ValueError(
                f"method {method.name!r} track lacks identity column {method.id_column!r}"
            )
    ffmpeg_path = Path(ffmpeg)
    if not ffmpeg_path.is_file() and shutil.which(ffmpeg) is None:
        raise FileNotFoundError(f"ffmpeg executable was not found: {ffmpeg}")


def render_episodes(
    *,
    experiment: str,
    episodes: Sequence[Episode],
    methods: Sequence[MethodSpec],
    slp_dir: Path,
    video_dir: Path,
    output_dir: Path,
    ffmpeg: str = DEFAULT_FFMPEG,
    python_executable: str = sys.executable,
    settings: RenderSettings = RenderSettings(),
    calibration: Path | None = None,
    identity_specs: Sequence[str] = (),
    node_names: Sequence[str] = ("mouth", "tail"),
    dry_run: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Render every episode for every method and write an auditable manifest."""

    if not episodes:
        raise ValueError("at least one episode is required")
    if not methods:
        raise ValueError("at least one method is required")
    output_dir = Path(output_dir)
    manifest_path = output_dir / "heldout_association_render_manifest.json"
    method_plans: list[dict[str, Any]] = []
    planned_artifacts: list[Path] = [manifest_path]
    for method in methods:
        method_dir = output_dir / method.slug
        clips = [
            method_dir
            / f"{_slug(episode.episode_id)}_f{episode.start_frame}-{episode.end_frame}.mp4"
            for episode in episodes
        ]
        concat_file = method_dir / "concat.txt"
        combined = output_dir / f"{method.slug}_heldout_episodes.mp4"
        planned_artifacts.extend([*clips, concat_file, combined])
        render_commands = [
            build_render_command(
                python_executable=python_executable, experiment=experiment,
                episode=episode, method=method, slp_dir=slp_dir, video_dir=video_dir,
                output_path=clip, ffmpeg=ffmpeg, settings=settings,
                calibration=calibration, identity_specs=identity_specs, node_names=node_names,
            )
            for episode, clip in zip(episodes, clips)
        ]
        method_plans.append(
            {
                "name": method.name, "slug": method.slug,
                "track_csv": str(method.track_csv), "id_column": method.id_column,
                "clips": [
                    {
                        "episode_id": episode.episode_id,
                        "source_start_frame": episode.start_frame,
                        "source_end_frame": episode.end_frame,
                        "source_n_frames": episode.n_frames,
                        "path": str(clip), "command": command, "status": "planned",
                    }
                    for episode, clip, command in zip(episodes, clips, render_commands)
                ],
                "concat_file": str(concat_file), "combined_video": str(combined),
                "concat_command": build_concat_command(ffmpeg, concat_file, combined),
                "status": "planned",
            }
        )

    existing = [path for path in planned_artifacts if path.exists()]
    if existing and not overwrite:
        preview = ", ".join(str(path) for path in existing[:3])
        suffix = " ..." if len(existing) > 3 else ""
        raise FileExistsError(f"planned output exists; pass --overwrite: {preview}{suffix}")
    if not dry_run:
        _assert_runtime_inputs(methods, slp_dir, video_dir, ffmpeg)

    manifest: dict[str, Any] = {
        "schema_version": "heldout_association_episode_render_v1",
        "status": "dry_run" if dry_run else "running",
        "dry_run": bool(dry_run), "experiment": experiment,
        "slp_dir": str(slp_dir), "video_dir": str(video_dir),
        "settings": asdict(settings),
        "overlay_policy": {
            "raw_sleap": "all raw instances in gray",
            "raw_by_index": False,
            "resolved_marker_arrow_thickness": settings.arrow_thickness,
        },
        "episodes": [asdict(episode) | {"n_frames": episode.n_frames} for episode in episodes],
        "methods": method_plans,
    }
    _write_manifest(manifest_path, manifest)
    if dry_run:
        return manifest

    try:
        for method_plan in method_plans:
            concat_paths: list[Path] = []
            for clip in method_plan["clips"]:
                clip_path = Path(clip["path"])
                clip_path.parent.mkdir(parents=True, exist_ok=True)
                completed = subprocess.run(clip["command"], check=False)
                if completed.returncode != 0 or not clip_path.is_file() or clip_path.stat().st_size == 0:
                    clip["status"] = "failed"
                    clip["returncode"] = int(completed.returncode)
                    raise RuntimeError(
                        f"render failed for {method_plan['name']} {clip['episode_id']} "
                        f"(return code {completed.returncode})"
                    )
                clip["status"] = "complete"
                clip["returncode"] = int(completed.returncode)
                clip["size_bytes"] = int(clip_path.stat().st_size)
                concat_paths.append(clip_path)
                _write_manifest(manifest_path, manifest)

            concat_file = Path(method_plan["concat_file"])
            concat_file.write_text(_ffconcat_text(concat_paths), encoding="utf-8")
            completed = subprocess.run(method_plan["concat_command"], check=False)
            if completed.returncode != 0:
                fallback = _build_reencode_command(
                    ffmpeg, concat_file, Path(method_plan["combined_video"]), settings.codec
                )
                method_plan["concat_fallback_command"] = fallback
                completed = subprocess.run(fallback, check=False)
            combined = Path(method_plan["combined_video"])
            if completed.returncode != 0 or not combined.is_file() or combined.stat().st_size == 0:
                method_plan["status"] = "failed"
                method_plan["returncode"] = int(completed.returncode)
                raise RuntimeError(
                    f"ffmpeg concat failed for method {method_plan['name']} "
                    f"(return code {completed.returncode})"
                )
            method_plan["status"] = "complete"
            method_plan["returncode"] = int(completed.returncode)
            method_plan["combined_size_bytes"] = int(combined.stat().st_size)
            _write_manifest(manifest_path, manifest)
        manifest["status"] = "complete"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        _write_manifest(manifest_path, manifest)
        raise
    _write_manifest(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment")
    parser.add_argument("--episodes-csv", type=Path, required=True)
    parser.add_argument("--slp-dir", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument(
        "--method", nargs=3, action="append", metavar=("NAME", "TRACK_CSV", "ID_COLUMN"),
        required=True, help="named track method; repeat to compare methods",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calib", type=Path)
    parser.add_argument("--identity", action="append", default=[], metavar="SOURCE=DISPLAY")
    parser.add_argument("--node-names", nargs="+", default=["mouth", "tail"])
    parser.add_argument("--panel", type=int, default=600)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--tail-seconds", type=float, default=5.0)
    parser.add_argument("--source-fps", type=float, default=166.0)
    parser.add_argument("--gate-px", type=float, default=80.0)
    parser.add_argument("--arrow-thickness", type=int, default=1)
    parser.add_argument("--codec", default="h264_nvenc")
    parser.add_argument("--nvenc-cq", type=int, default=24)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--ffmpeg", default=DEFAULT_FFMPEG)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    episodes = load_episodes(args.episodes_csv)
    methods = parse_methods(args.method)
    settings = RenderSettings(
        panel=args.panel, fps=args.fps, frame_step=args.frame_step,
        tail_seconds=args.tail_seconds, source_fps=args.source_fps, gate_px=args.gate_px,
        arrow_thickness=args.arrow_thickness, codec=args.codec,
        nvenc_cq=args.nvenc_cq, workers=args.workers,
    )
    manifest = render_episodes(
        experiment=args.experiment, episodes=episodes, methods=methods,
        slp_dir=args.slp_dir, video_dir=args.video_dir, output_dir=args.output_dir,
        ffmpeg=args.ffmpeg, python_executable=args.python_executable, settings=settings,
        calibration=args.calib, identity_specs=args.identity, node_names=args.node_names,
        dry_run=args.dry_run, overwrite=args.overwrite,
    )
    print(json.dumps({"status": manifest["status"], "methods": len(methods),
                      "episodes": len(episodes),
                      "manifest": str(args.output_dir / "heldout_association_render_manifest.json")},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
