from __future__ import annotations

"""Atomically repair an existing audit bundle to browser-compatible H.264."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from .render_blinded_audit import transcode_browser_mp4


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_video(path: Path) -> dict[str, object]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("ffprobe is required to verify repaired audit clips")
    command = [
        ffprobe,
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,profile,pix_fmt,width,height,avg_frame_rate,nb_read_frames",
        "-of",
        "json",
        str(path),
    ]
    payload = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
    if len(payload.get("streams", [])) != 1:
        raise RuntimeError(f"expected one video stream in {path}")
    stream = payload["streams"][0]
    numerator, denominator = str(stream["avg_frame_rate"]).split("/", maxsplit=1)
    return {
        "codec": stream["codec_name"],
        "profile": stream.get("profile"),
        "pixel_format": stream.get("pix_fmt"),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": float(numerator) / float(denominator),
        "frames": int(stream["nb_read_frames"]),
    }


def repair_clip(path: Path, *, output_fps: int) -> dict[str, object]:
    before = probe_video(path)
    before_sha256 = _sha256(path)
    if (
        before["codec"] == "h264"
        and before["pixel_format"] == "yuv420p"
        and abs(float(before["fps"]) - output_fps) < 1e-6
    ):
        return {
            "clip": path.name,
            "action": "already_compatible",
            "before": before,
            "after": before,
            "before_sha256": before_sha256,
            "after_sha256": before_sha256,
        }

    candidate = path.with_name(f".{path.stem}.browser_candidate.mp4")
    candidate.unlink(missing_ok=True)
    try:
        transcode_browser_mp4(path, candidate, output_fps=output_fps)
        after = probe_video(candidate)
        mismatches = []
        if after["codec"] != "h264":
            mismatches.append(f"codec={after['codec']}")
        if after["pixel_format"] != "yuv420p":
            mismatches.append(f"pixel_format={after['pixel_format']}")
        if after["frames"] != before["frames"]:
            mismatches.append(f"frames={after['frames']} expected={before['frames']}")
        if (after["width"], after["height"]) != (before["width"], before["height"]):
            mismatches.append(
                f"size={after['width']}x{after['height']} expected={before['width']}x{before['height']}"
            )
        if abs(float(after["fps"]) - output_fps) > 1e-6:
            mismatches.append(f"fps={after['fps']} expected={output_fps}")
        if mismatches:
            raise RuntimeError(f"verification failed for {path.name}: {', '.join(mismatches)}")
        after_sha256 = _sha256(candidate)
        candidate.replace(path)
        return {
            "clip": path.name,
            "action": "transcoded",
            "before": before,
            "after": after,
            "before_sha256": before_sha256,
            "after_sha256": after_sha256,
        }
    finally:
        candidate.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-dir", type=Path, required=True)
    parser.add_argument("--output-fps", type=int, default=60)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    clips = sorted(args.clip_dir.glob("*.mp4"))
    if not clips:
        raise ValueError(f"no MP4 clips found under {args.clip_dir}")
    rows = [repair_clip(path, output_fps=int(args.output_fps)) for path in clips]
    report = {
        "schema_version": "audit_clip_codec_repair_v1",
        "clip_dir": str(args.clip_dir.resolve()),
        "clips": len(rows),
        "transcoded": sum(row["action"] == "transcoded" for row in rows),
        "already_compatible": sum(row["action"] == "already_compatible" for row in rows),
        "required_codec": "h264",
        "required_pixel_format": "yuv420p",
        "output_fps": int(args.output_fps),
        "all_frame_counts_preserved": all(
            row["before"]["frames"] == row["after"]["frames"] for row in rows
        ),
        "files": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    print(json.dumps(run(build_parser().parse_args(argv)), indent=2))


if __name__ == "__main__":
    main()
