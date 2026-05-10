from __future__ import annotations

"""
Expand a multi-view annotation seed so every (experiment, frame_idx) group
has all three views present and their frames embedded in the file.

Why
---
The round-2/round-3 seed has 500 single-view LFs (only one view per
``(experiment, frame_idx)`` pair) plus 499 multi-view LFs. When the
multi-view annotator opens a single-view group it falls back to reading
the two missing-view frames from ``medaka_data_dir`` (the Z:\\ network
drive). That works on the lab PC but breaks for a colleague who only has
the bundled folder.

This tool fills the gaps:

1. Load the existing seed (frames embedded for the views already labelled).
2. For each ``(experiment, frame_idx)`` group with fewer than 3 views,
   add an *empty* ``LabeledFrame`` for each missing view, pointing at the
   matching source video on ``medaka_data_dir``.
3. Re-save with ``embed="all"`` so every LF (existing + newly-added empty
   ones) has its frame stored inside the new pkg.slp.

The result is a fully self-contained seed: a colleague can navigate every
group and see all three camera views even without access to Z:\\.

Usage
-----
    python -m scripts.sleap.expand_seed_with_missing_views \\
        --seed  C:/Users/.../annotation_seed_round3.pkg.slp \\
        --paths-config configs/paths.local.toml \\
        --out   C:/Users/.../annotation_seed_round3_expanded.pkg.slp
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.common.local_paths import choose_path, load_local_paths


VIEWS: Tuple[str, ...] = ("top_cam", "side1_cam", "side2_cam")
VIEW_TO_VIDEO = {
    "top_cam": "cam0_top.mp4",
    "side1_cam": "cam1_side.mp4",
    "side2_cam": "cam2_side.mp4",
}


def _video_source_path(video) -> str:
    """Walk the ``source_video`` chain on a sleap-io Video and return the
    deepest path that points at an actual on-disk video file. See the
    annotator's helper of the same name for full notes.
    """
    video_exts = (".mp4", ".avi", ".mov", ".mkv", ".webm")
    visited: set[int] = set()
    cur = video
    while cur is not None and id(cur) not in visited:
        visited.add(id(cur))
        backend = getattr(cur, "backend", None)
        if backend is not None:
            sf = getattr(backend, "source_filename", None)
            if sf and str(sf).lower().endswith(video_exts):
                return str(sf)
        fname = getattr(cur, "filename", None)
        if fname and str(fname).lower().endswith(video_exts):
            return str(fname)
        cur = getattr(cur, "source_video", None)
    fallback_backend = getattr(video, "backend", None)
    fb = getattr(fallback_backend, "source_filename", None) if fallback_backend else None
    if fb:
        return str(fb)
    return str(getattr(video, "filename", "")) or ""


def _classify_view(fname: str) -> Optional[str]:
    s = str(fname).lower().replace("\\", "/")
    for tag in VIEWS:
        if f"/{tag}/" in s:
            return tag
    if "side1" in s:
        return "side1_cam"
    if "side2" in s:
        return "side2_cam"
    if "top_cam" in s or "cam0_top" in s:
        return "top_cam"
    return None


def _experiment_from_video(fname: str) -> Optional[str]:
    parts = Path(str(fname).replace("\\", "/")).parts
    for i, p in enumerate(parts):
        if p in VIEWS and i > 0:
            return parts[i - 1]
    return None


def expand(seed_path: Path, out_path: Path, medaka_data_dir: Path) -> dict:
    import sleap_io as sio

    print(f"Loading {seed_path} …")
    src = sio.load_slp(str(seed_path))
    print(
        f"  {len(src.labeled_frames)} LFs, {len(src.videos)} videos, "
        f"{len(src.suggestions)} suggestions."
    )

    # Map (experiment, view) -> on-disk MediaVideo for *new* LFs at frame
    # indices that aren't already embedded. The seed's HDF5-backed Videos
    # only have specific frames embedded; ``HDF5Video.get_frame`` errors
    # for any other index. Using the underlying MediaVideo lets
    # ``sio.save_slp(..., embed="all")`` re-read those frames from the
    # source mp4 and embed them into the new pkg.slp.
    exp_view_to_video: Dict[Tuple[str, str], "sio.Video"] = {}
    for v in src.videos:
        s_path = _video_source_path(v)
        view = _classify_view(s_path)
        exp = _experiment_from_video(s_path)
        if view is None or exp is None:
            continue
        # Walk to a non-embedded source. ``v`` may be HDF5; its
        # ``source_video`` chain ends at a MediaVideo whose
        # ``backend.source_filename`` is the mp4 path.
        cur = v
        for _ in range(8):
            sv = getattr(cur, "source_video", None)
            if sv is None:
                break
            cur = sv
        exp_view_to_video.setdefault((exp, view), cur)

    # Group existing LFs.
    existing: Dict[Tuple[str, int], Dict[str, "sio.LabeledFrame"]] = defaultdict(dict)
    for lf in src.labeled_frames:
        s_path = _video_source_path(lf.video)
        view = _classify_view(s_path)
        exp = _experiment_from_video(s_path)
        if view is None or exp is None:
            continue
        existing[(exp, int(lf.frame_idx))][view] = lf

    # Build the augmented LF list, reusing existing Video objects.
    new_lfs = list(src.labeled_frames)
    added = 0
    no_video = 0
    pre_hist = defaultdict(int)
    for (exp, frame_idx), view_lfs in existing.items():
        pre_hist[len(view_lfs)] += 1
        missing = [v for v in VIEWS if v not in view_lfs]
        for view in missing:
            video = exp_view_to_video.get((exp, view))
            if video is None:
                # No existing Video for this (exp, view) — fall back to
                # opening the source mp4 from medaka_data_dir.
                disk = medaka_data_dir / exp / view / VIEW_TO_VIDEO[view]
                if not disk.exists():
                    no_video += 1
                    continue
                try:
                    video = sio.load_video(str(disk))
                    exp_view_to_video[(exp, view)] = video
                except Exception:
                    no_video += 1
                    continue
            new_lfs.append(sio.LabeledFrame(
                video=video, frame_idx=int(frame_idx), instances=[],
            ))
            added += 1

    print(f"  views-per-group before: {dict(pre_hist)}")
    print(f"  added {added} empty LF(s) for missing views (no_video={no_video}).")

    # Dedupe video list by id; one per (exp, view) by construction.
    new_videos = list({id(v): v for v in (lf.video for lf in new_lfs)}.values())
    suggestions = [
        sio.SuggestionFrame(video=lf.video, frame_idx=int(lf.frame_idx))
        for lf in new_lfs
    ]
    new_labels = sio.Labels(
        skeletons=src.skeletons,
        videos=new_videos,
        labeled_frames=new_lfs,
        suggestions=suggestions,
        provenance=src.provenance,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Saving {len(new_lfs)} LFs to {out_path} with embed='all' …"
        " (this re-extracts each frame, expect a few minutes for ~2k LFs)"
    )
    sio.save_slp(new_labels, str(out_path), embed="all")
    size_mb = out_path.stat().st_size / 1e6
    print(f"  done. {size_mb:.1f} MB")

    return {
        "lfs_in": len(src.labeled_frames),
        "lfs_out": len(new_lfs),
        "added": added,
        "no_video": no_video,
        "out_path": str(out_path),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--paths-config", default=None)
    ap.add_argument("--medaka-data-dir", default=None)
    args = ap.parse_args(argv)

    local = load_local_paths(args.paths_config)
    medaka_data_dir = choose_path(
        args.medaka_data_dir, local.medaka_data_dir if local else None,
    )
    if medaka_data_dir is None:
        ap.error("medaka_data_dir not set (paths.local.toml or --medaka-data-dir).")

    summary = expand(args.seed, args.out, Path(medaka_data_dir))
    print(summary)


if __name__ == "__main__":
    main()
