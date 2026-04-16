"""Resolve the cam0 session mp4 and clip-to-session frame offset for a set."""
from __future__ import annotations

import re
from pathlib import Path

from scripts.tracking.missing_value_filling import extract_time_from_filename

# Parses the experiment-folder prefix out of a set key, e.g.
#   20251225_K_1_2_FCMC_m1tf_Courtship_290.729-296.331
#   -> 20251225_K_1_2_FCMC
_EXPERIMENT_PREFIX_RE = re.compile(
    r"^(?P<folder>\d{8}_[A-Za-z]+_\d+_\d+_\w+?)_"
)


def experiment_folder_from_set_key(set_key: str) -> str | None:
    m = _EXPERIMENT_PREFIX_RE.match(set_key)
    return m.group("folder") if m else None


def resolve_cam0_video_and_offset(
    file_set_info: dict,
    medaka_data_dir: str | Path,
    fps: int = 166,
) -> tuple[Path, int] | None:
    """
    Return `(session_cam0_mp4_path, frame_offset)` for this clip, or None.

    The clip CSV's `frame` column is 0-based within the clip. To read frames
    from the full-session cam0 mp4 we add `frame_offset = clip_start_sec * fps`.
    """
    set_key = file_set_info.get("key")
    player1_csv = file_set_info.get("player1_file")
    if set_key is None or player1_csv is None:
        return None

    folder = experiment_folder_from_set_key(set_key)
    if folder is None:
        return None

    video_path = Path(medaka_data_dir) / folder / "top_cam" / "cam0_top.mp4"
    if not video_path.exists():
        return None

    start_sec, _ = extract_time_from_filename(player1_csv)
    if start_sec is None:
        return None

    return video_path, int(round(start_sec * fps))
