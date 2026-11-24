from __future__ import annotations

"""
Merge per-view SLEAP chunks produced by the HPC pipeline into a single .slp
per view and update the canonical experiment manifest.

The manifest is always the raw-recording one:
    <medaka_data_dir>/<experiment>/experiment_metadata.json

This script reads it (or creates it from the folder name + each camera's
metadata.csv if it does not yet exist), attaches a `sleap` section describing
the chunk layout and merge status, and writes it back. Hand-authored fields
(e.g. `animal`) are preserved.

Merged .slp files stay next to the chunks:
    <sleap_output_dir>/<experiment>/<sleap_subdir>/<view>.slp

`sleap_subdir` is optional for backward compatibility. It is intended for
round tags such as `round8_260729_snn031_m6`.

If the merged .slp is already present and newer than every chunk, the merge
itself is skipped; the manifest is still refreshed.
"""

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.common.experiment_metadata import (
    VIEWS,
    load_metadata_json,
    manifest_path_for,
    write_experiment_manifest,
)
from scripts.common.local_paths import choose_path, load_local_paths


CHUNK_RE = re.compile(r"^(?P<view>\w+)\.frames_(?P<start>\d+)-(?P<end>\d+)\.slp$")


@dataclass
class ViewMerge:
    """Record of one view's merge status (written into the metadata JSON)."""

    view: str
    chunks_dir: str
    chunks: list[str] = field(default_factory=list)
    frame_ranges: list[tuple[int, int]] = field(default_factory=list)
    merged_path: str | None = None
    total_frames: int | None = None
    labeled_frames: int | None = None
    merged: bool = False
    skipped_reason: str | None = None


def find_chunks(chunks_dir: Path, view: str) -> list[Path]:
    """Return chunk .slp paths for a view, sorted by start frame."""
    candidates: list[tuple[int, Path]] = []
    for path in chunks_dir.glob(f"{view}.frames_*.slp"):
        m = CHUNK_RE.match(path.name)
        if not m or m["view"] != view:
            continue
        candidates.append((int(m["start"]), path))
    return [p for _, p in sorted(candidates, key=lambda t: t[0])]


def _merged_is_fresh(merged_path: Path, chunks: list[Path]) -> bool:
    if not merged_path.exists() or merged_path.stat().st_size == 0:
        return False
    merged_mtime = merged_path.stat().st_mtime
    return all(chunk.stat().st_mtime <= merged_mtime for chunk in chunks)


def validate_chunk_ranges(
    chunk_paths: list[Path],
    view: str,
    expected_total_frames: int | None = None,
) -> list[tuple[int, int]]:
    """Validate and return contiguous, non-overlapping inclusive frame ranges."""
    if not chunk_paths:
        raise ValueError(f"[{view}] no chunk files provided")

    ranges: list[tuple[int, int]] = []
    for path in chunk_paths:
        match = CHUNK_RE.match(path.name)
        if not match or match["view"] != view:
            raise ValueError(f"[{view}] invalid chunk filename: {path.name}")
        start, end = int(match["start"]), int(match["end"])
        if end < start:
            raise ValueError(f"[{view}] invalid chunk range {start}-{end}")
        ranges.append((start, end))

    if ranges[0][0] != 0:
        raise ValueError(f"[{view}] chunk coverage starts at {ranges[0][0]}, expected 0")
    for previous, current in zip(ranges, ranges[1:]):
        expected_start = previous[1] + 1
        if current[0] != expected_start:
            relation = "overlap" if current[0] <= previous[1] else "gap"
            raise ValueError(
                f"[{view}] chunk {relation}: {previous[0]}-{previous[1]} then "
                f"{current[0]}-{current[1]} (expected next start {expected_start})"
            )

    if expected_total_frames is not None:
        expected_end = expected_total_frames - 1
        if ranges[-1][1] != expected_end:
            raise ValueError(
                f"[{view}] chunk coverage ends at {ranges[-1][1]}, "
                f"expected {expected_end}"
            )
    return ranges


def count_labeled_frames(path: Path) -> int:
    """Load one SLP and return its labeled-frame count."""
    try:
        import sleap_io as sio
    except ImportError as exc:
        raise RuntimeError("sleap-io is required to inspect merged SLP files") from exc
    return len(sio.load_slp(str(path)).labeled_frames)


def merge_view_chunks(
    chunk_paths: list[Path],
    output_path: Path,
    canonical_video_path: str | None = None,
) -> int:
    """
    Concatenate chunk .slp files into `output_path` using sleap-io.

    The merged file is normalised so that all frames reference a single
    Video and a single Skeleton (the first chunk's). If
    `canonical_video_path` is supplied, the merged Video's filename is
    rewritten to that path — useful for replacing the staged /work path
    with the permanent /bucket path.

    Returns the number of labeled frames written.
    """
    try:
        import sleap_io as sio
    except ImportError as exc:
        raise RuntimeError(
            "sleap-io is required to merge chunks. Install with:\n"
            "    uv sync --extra sleap\n"
            "or  pip install sleap-io"
        ) from exc

    if not chunk_paths:
        raise ValueError("No chunk files provided")

    parts = [sio.load_slp(str(p)) for p in chunk_paths]

    canonical_video = parts[0].videos[0]
    if canonical_video_path:
        canonical_video.replace_filename(canonical_video_path, open=False)
    canonical_skeleton = parts[0].skeletons[0]

    frames = []
    for lbls in parts:
        for f in lbls.labeled_frames:
            f.video = canonical_video
            for inst in f.instances:
                inst.skeleton = canonical_skeleton
            frames.append(f)

    merged = sio.Labels(
        labeled_frames=frames,
        videos=[canonical_video],
        skeletons=[canonical_skeleton],
        tracks=parts[0].tracks,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Save under a temp name and rename: a half-written merged file would be
    # newer than every chunk and so pass _merged_is_fresh on the next run.
    tmp_path = output_path.with_name(output_path.stem + ".merging.slp")
    sio.save_slp(merged, str(tmp_path))
    tmp_path.replace(output_path)
    return len(frames)


def merge_experiment(
    experiment_name: str,
    sleap_output_dir: str | Path,
    raw_data_dir: str | Path,
    sleap_subdir: str | None = None,
    force: bool = False,
    views: tuple[str, ...] = VIEWS,
) -> dict:
    """
    Merge all views for one experiment and augment the raw-folder manifest
    with a `sleap` section.

    Parameters
    ----------
    experiment_name : str
        Folder name like `20251220_D_1_1_FCMC`.
    sleap_output_dir : path-like
        Root holding `<experiment_name>/` with chunk .slp files.
    raw_data_dir : path-like
        Root holding the raw recordings. The canonical manifest lives at
        `<raw_data_dir>/<experiment_name>/experiment_metadata.json`.
    force : bool
        Re-merge even if the merged file is already fresh.
    views : tuple of str
        Views to process; default top_cam, side1_cam, side2_cam.

    Returns
    -------
    dict
        The final manifest contents (with the updated `sleap` section).
    """
    sleap_output_dir = Path(sleap_output_dir)
    chunks_dir = sleap_output_dir / experiment_name
    if sleap_subdir:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", sleap_subdir):
            raise ValueError("sleap_subdir must be one path component")
        chunks_dir = chunks_dir / sleap_subdir
    if not chunks_dir.is_dir():
        raise FileNotFoundError(f"Experiment chunks dir not found: {chunks_dir}")

    raw_data_dir = Path(raw_data_dir)
    raw_exp_dir = raw_data_dir / experiment_name
    if not raw_exp_dir.is_dir():
        raise FileNotFoundError(
            f"Raw recording folder not found: {raw_exp_dir}. "
            "The manifest lives in the raw folder, so it must exist."
        )

    # Load (or create) the raw-folder manifest so we can read canonical video
    # paths before merging. Passing sleap_section=None leaves any existing
    # `sleap` key untouched for now; we'll rewrite it after merging.
    manifest = write_experiment_manifest(raw_exp_dir)
    view_records = manifest.get("views", {})

    merge_status: dict[str, ViewMerge] = {}
    for view in views:
        status = ViewMerge(view=view, chunks_dir=str(chunks_dir))
        chunks = find_chunks(chunks_dir, view)
        if not chunks:
            status.skipped_reason = "no chunks found"
            merge_status[view] = status
            continue

        status.chunks = [p.name for p in chunks]
        view_record = view_records.get(view) or {}
        expected_total_frames = view_record.get("total_frames")
        status.frame_ranges = validate_chunk_ranges(
            chunks,
            view,
            expected_total_frames=expected_total_frames,
        )
        status.total_frames = (
            expected_total_frames
            if expected_total_frames is not None
            else status.frame_ranges[-1][1] + 1
        )

        merged_path = chunks_dir / f"{view}.slp"
        status.merged_path = str(merged_path)

        if not force and _merged_is_fresh(merged_path, chunks):
            status.merged = True
            status.skipped_reason = "merged file already fresh"
            status.labeled_frames = count_labeled_frames(merged_path)
            merge_status[view] = status
            print(f"[{view}] skip — {merged_path.name} is newer than all chunks")
            continue

        print(f"[{view}] merging {len(chunks)} chunks -> {merged_path.name}")
        canonical_video_path = (view_records.get(view) or {}).get("video_path")
        n_frames = merge_view_chunks(
            chunks, merged_path, canonical_video_path=canonical_video_path
        )
        status.merged = True
        status.labeled_frames = n_frames
        merge_status[view] = status
        print(f"[{view}] wrote {n_frames} labeled frames")

    sleap_section = {
        "chunks_dir": str(chunks_dir),
        "sleap_subdir": sleap_subdir,
        "views": {name: asdict(status) for name, status in merge_status.items()},
    }

    final_manifest = write_experiment_manifest(
        raw_exp_dir, sleap_section=sleap_section
    )
    print(f"Updated manifest: {manifest_path_for(raw_exp_dir)}")
    return final_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge per-view SLEAP chunk files for one medaka experiment into "
            "a single .slp per view and write experiment_metadata.json."
        )
    )
    parser.add_argument(
        "experiment",
        help="Experiment folder name, e.g. 20251220_D_1_1_FCMC.",
    )
    parser.add_argument(
        "--paths-config",
        default=None,
        help="TOML paths config; defaults to configs/paths.local.toml if present.",
    )
    parser.add_argument(
        "--sleap-output-dir",
        default=None,
        help="Override sleap_output_dir from paths.local.toml.",
    )
    parser.add_argument(
        "--medaka-data-dir",
        default=None,
        help="Override medaka_data_dir from paths.local.toml (for reading metadata.csv).",
    )
    parser.add_argument(
        "--sleap-subdir",
        "--round-tag",
        dest="sleap_subdir",
        default=None,
        help=(
            "Optional round-tag subdirectory below "
            "sleap_output_dir/<experiment>/, e.g. round8_260729_snn031_m6."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-merge even if the merged .slp is already fresh.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    local = load_local_paths(args.paths_config)
    sleap_output_dir = choose_path(
        args.sleap_output_dir,
        local.sleap_output_dir if local else None,
    )
    medaka_data_dir = choose_path(
        args.medaka_data_dir,
        local.medaka_data_dir if local else None,
    )

    if sleap_output_dir is None:
        parser.error(
            "sleap_output_dir is required. Pass --sleap-output-dir or set it "
            "in configs/paths.local.toml."
        )
    if medaka_data_dir is None:
        parser.error(
            "medaka_data_dir is required (the manifest lives in the raw "
            "recording folder). Pass --medaka-data-dir or set medaka_data_dir "
            "in configs/paths.local.toml."
        )

    merge_experiment(
        experiment_name=args.experiment,
        sleap_output_dir=sleap_output_dir,
        raw_data_dir=medaka_data_dir,
        sleap_subdir=args.sleap_subdir,
        force=args.force,
    )


if __name__ == "__main__":
    main()
