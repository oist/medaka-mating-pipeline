from __future__ import annotations

"""Render method-blinded crossing clips and an independent audit worksheet."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .build_audit_player import write_audit_player


PALETTE = (
    (46, 204, 113),
    (52, 152, 219),
    (231, 76, 60),
    (155, 89, 182),
    (241, 196, 15),
    (26, 188, 156),
)


def _track_label_map(table: pd.DataFrame) -> dict[str, str]:
    ids = sorted(table["track_id"].astype(str).unique())
    return {track_id: f"T{index}" for index, track_id in enumerate(ids, start=1)}


def _track_color_map(table: pd.DataFrame) -> dict[str, tuple[int, int, int]]:
    ids = sorted(table["track_id"].astype(str).unique())
    colors = {}
    for index, track_id in enumerate(ids):
        rgb = PALETTE[index % len(PALETTE)]
        colors[track_id] = rgb[2], rgb[1], rgb[0]
    return colors


def _read_tracks(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    required = {"frame", "track_id", "x", "y"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{path} missing {sorted(missing)}")
    out = table[["frame", "track_id", "x", "y"]].copy()
    out["frame"] = pd.to_numeric(out["frame"], errors="raise").astype(int)
    out["track_id"] = out["track_id"].astype(str)
    return out[np.isfinite(out[["x", "y"]]).all(axis=1)]


def _draw_label(frame: np.ndarray, x: float, y: float, label: str, color) -> None:
    center = int(round(x)), int(round(y))
    cv2.circle(frame, center, 18, color, 4, cv2.LINE_AA)
    origin = center[0] + 22, center[1] - 12
    cv2.putText(frame, label, origin, cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(frame, label, origin, cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2, cv2.LINE_AA)


def transcode_browser_mp4(
    source: Path,
    destination: Path,
    *,
    output_fps: int = 60,
) -> None:
    """Transcode an intermediate OpenCV MP4 to browser-compatible H.264."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to create browser-compatible H.264 audit clips")
    temporary = destination.with_name(f".{destination.stem}.h264_tmp.mp4")
    temporary.unlink(missing_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-an",
        "-vf",
        f"setpts=N/({int(output_fps)}*TB)",
        "-r",
        str(int(output_fps)),
        "-fps_mode",
        "cfr",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-profile:v",
        "high",
        "-level:v",
        "5.1",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    try:
        subprocess.run(command, check=True)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def render_clip(
    *,
    video_path: Path,
    output_path: Path,
    rows_by_frame: dict[int, pd.DataFrame],
    labels: dict[str, str],
    colors: dict[str, tuple[int, int, int]],
    absolute_video_frame_zero: int,
    clip_start: int,
    clip_end: int,
    event_start: int,
    event_end: int,
    event_id: str,
    blind_code: str,
    close_intervals: list[tuple[int, int]] | None = None,
    output_fps: int = 60,
) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    relative_start = clip_start - int(absolute_video_frame_zero)
    if relative_start < 0:
        raise ValueError("clip starts before video frame zero")
    capture.set(cv2.CAP_PROP_POS_FRAMES, relative_start)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    intermediate = output_path.with_name(f".{output_path.stem}.mpeg4_intermediate.mp4")
    intermediate.unlink(missing_ok=True)
    writer = cv2.VideoWriter(
        str(intermediate), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not create {output_path}")
    frames_written = 0
    try:
        for absolute_frame in range(clip_start, clip_end + 1):
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(
                    f"video ended while rendering {event_id} at absolute frame {absolute_frame}"
                )
            rows = rows_by_frame.get(absolute_frame)
            if rows is not None:
                for _, row in rows.iterrows():
                    track_id = str(row["track_id"])
                    _draw_label(
                        image,
                        float(row["x"]),
                        float(row["y"]),
                        labels.get(track_id, "T?"),
                        colors[track_id],
                    )
            intervals = close_intervals or [(event_start, event_end)]
            in_crossing = any(start <= absolute_frame <= end for start, end in intervals)
            header = f"{event_id}   BLINDED METHOD {blind_code}"
            cv2.rectangle(image, (0, 0), (width, 62), (20, 20, 20), -1)
            cv2.putText(
                image, header, (22, 40), cv2.FONT_HERSHEY_DUPLEX, 0.95,
                (245, 245, 245), 2, cv2.LINE_AA,
            )
            if in_crossing:
                cv2.rectangle(image, (4, 4), (width - 5, height - 5), (0, 165, 255), 8)
                cv2.putText(
                    image, "CLOSE APPROACH", (width - 310, 40),
                    cv2.FONT_HERSHEY_DUPLEX, 0.72, (0, 190, 255), 2, cv2.LINE_AA,
                )
            writer.write(image)
            frames_written += 1
    finally:
        writer.release()
        capture.release()
    expected_frames = clip_end - clip_start + 1
    if frames_written != expected_frames:
        raise RuntimeError(
            f"rendered {frames_written}/{expected_frames} frames for {event_id}"
        )
    try:
        transcode_browser_mp4(intermediate, output_path, output_fps=output_fps)
    finally:
        intermediate.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--video-frame-zero", type=int, default=0)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--method", action="append", nargs=2, metavar=("NAME", "TRACKS_CSV"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context-frames", type=int, default=83)
    parser.add_argument("--clip-fps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument(
        "--reuse-existing-clips",
        action="store_true",
        help="Reuse already-rendered clips while rebuilding the worksheet/player metadata.",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    events = pd.read_csv(args.events)
    required_events = {"event_id", "start_frame", "end_frame"}
    if missing := required_events - set(events.columns):
        raise ValueError(f"events table missing {sorted(missing)}")
    explicit_clip_bounds = {"clip_start_frame", "clip_end_frame"} <= set(events.columns)
    method_tables = {name: _read_tracks(Path(path)) for name, path in args.method}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    clip_dir = args.output_dir / "clips"
    rng = np.random.default_rng(args.seed)
    audit_rows: list[dict[str, object]] = []
    key_rows: list[dict[str, object]] = []
    reused_clips = 0
    for _, event in events.sort_values("event_id").iterrows():
        method_names = list(method_tables)
        rng.shuffle(method_names)
        for blind_index, method_name in enumerate(method_names):
            blind_code = chr(ord("A") + blind_index)
            table = method_tables[method_name]
            rows_by_frame = {
                int(frame): group for frame, group in table.groupby("frame", sort=False)
            }
            labels = _track_label_map(table)
            colors = _track_color_map(table)
            event_id = str(event["event_id"])
            clip_name = f"{event_id}_method_{blind_code}.mp4"
            if explicit_clip_bounds:
                clip_start = int(event["clip_start_frame"])
                clip_end = int(event["clip_end_frame"])
            else:
                clip_start = int(event["start_frame"]) - int(args.context_frames)
                clip_end = int(event["end_frame"]) + int(args.context_frames)
            close_intervals = None
            if "close_intervals_json" in events.columns and pd.notna(event["close_intervals_json"]):
                parsed = json.loads(str(event["close_intervals_json"]))
                close_intervals = [
                    (int(interval["start_frame"]), int(interval["end_frame"]))
                    for interval in parsed
                ]
            output_path = clip_dir / clip_name
            if bool(args.reuse_existing_clips) and output_path.exists():
                reused_clips += 1
            else:
                render_clip(
                    video_path=args.video,
                    output_path=output_path,
                    rows_by_frame=rows_by_frame,
                    labels=labels,
                    colors=colors,
                    absolute_video_frame_zero=args.video_frame_zero,
                    clip_start=clip_start,
                    clip_end=clip_end,
                    event_start=int(event["start_frame"]),
                    event_end=int(event["end_frame"]),
                    event_id=event_id,
                    blind_code=blind_code,
                    close_intervals=close_intervals,
                    output_fps=int(args.clip_fps),
                )
            audit_rows.append(
                {
                    "event_id": event_id,
                    "blinded_method": blind_code,
                    "clip": f"clips/{clip_name}",
                    "continuity": "",
                    "localization": "",
                    "confidence_1_to_3": "",
                    "notes": "",
                }
            )
            key_rows.append(
                {
                    "event_id": event_id,
                    "blinded_method": blind_code,
                    "method": method_name,
                    "clip": f"clips/{clip_name}",
                }
            )
    worksheet = pd.DataFrame(audit_rows)
    worksheet_path = args.output_dir / "AUDIT_WORKSHEET.csv"
    worksheet.to_csv(worksheet_path, index=False)
    pd.DataFrame(key_rows).to_csv(args.output_dir / "PRIVATE_BLINDING_KEY.csv", index=False)
    write_audit_player(
        worksheet_path,
        args.output_dir / "AUDIT_PLAYER.html",
        seed=int(args.seed) + 1,
    )
    instructions = "# Blinded crossing audit\n\n"
    instructions += "Keep `PRIVATE_BLINDING_KEY.csv` closed. Open `AUDIT_PLAYER.html`, rate each clip, and download the completed CSV. You may instead fill in `AUDIT_WORKSHEET.csv` directly.\n\n"
    instructions += "- `continuity`: `pass` if every visible T-label stays on the same fish across the close approach; `switch` if any label changes fish; `uncertain` if you cannot tell.\n"
    instructions += "- `localization`: `pass` if all three labels are on fish before and after; `fail` if labels are missing or visibly off-fish; `uncertain` if ambiguous.\n"
    instructions += "- `confidence_1_to_3`: 1 low, 2 medium, 3 high.\n\n"
    instructions += "The orange border marks the automatically selected close-approach interval(s). Judge identity by comparing the clearly separated fish at the beginning and end. Method letters are randomized separately for every encounter.\n"
    instructions += f"Clips use browser-compatible H.264 and preserve every source frame at {int(args.clip_fps)} fps for easier inspection.\n"
    if "source_event_ids" in events.columns:
        source_fragments = sorted(
            {
                item
                for value in events["source_event_ids"].astype(str)
                for item in value.split(";")
                if item
            }
        )
        instructions += (
            f"\nThis revised bundle consolidates {len(source_fragments)} short detector fragments "
            f"into {len(events)} separation-delimited encounters. It supersedes the short-context "
            "ratings for biological identity continuity; keep the earlier ratings only as preliminary evidence.\n"
        )
    if {"separation_threshold_px", "separation_hold_frames"} <= set(events.columns):
        threshold = float(events["separation_threshold_px"].iloc[0])
        hold_frames = int(events["separation_hold_frames"].iloc[0])
        seconds_text = ""
        if "separation_hold_seconds" in events.columns:
            seconds_text = f" ({float(events['separation_hold_seconds'].iloc[0]):g} source second)"
        instructions += (
            f"Every clip begins and ends with the crossing pair at least {threshold:g} pixels apart "
            f"for {hold_frames} consecutive source frames{seconds_text}.\n"
        )
    (args.output_dir / "README.md").write_text(instructions, encoding="utf-8")
    summary = {
        "schema_version": "blinded_crossing_audit_v1",
        "video": str(args.video.resolve()),
        "video_frame_zero": int(args.video_frame_zero),
        "events": int(len(events)),
        "methods": int(len(method_tables)),
        "clips": int(len(audit_rows)),
        "reused_existing_clips": int(reused_clips),
        "context_mode": "explicit_per_event_bounds" if explicit_clip_bounds else "fixed_symmetric",
        "context_frames": None if explicit_clip_bounds else int(args.context_frames),
        "randomization_seed": int(args.seed),
        "presentation_seed": int(args.seed) + 1,
        "audit_player": "AUDIT_PLAYER.html",
        "clip_codec": "H.264/AVC (libx264, yuv420p)",
        "clip_fps": int(args.clip_fps),
        "audit_status": "awaiting independent human ratings",
    }
    if "source_event_ids" in events.columns:
        summary["source_event_fragments"] = int(
            len(
                {
                    item
                    for value in events["source_event_ids"].astype(str)
                    for item in value.split(";")
                    if item
                }
            )
        )
        summary["supersedes"] = "../blinded_human_audit short-context continuity ratings"
    if {"separation_threshold_px", "separation_hold_frames"} <= set(events.columns):
        summary["separation_threshold_px"] = float(events["separation_threshold_px"].iloc[0])
        summary["separation_hold_frames"] = int(events["separation_hold_frames"].iloc[0])
        if "separation_hold_seconds" in events.columns:
            summary["separation_hold_seconds"] = float(events["separation_hold_seconds"].iloc[0])
    (args.output_dir / "audit_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    print(json.dumps(run(build_parser().parse_args(argv)), indent=2))


if __name__ == "__main__":
    main()
