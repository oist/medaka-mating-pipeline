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

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.common.local_paths import choose_path, load_local_paths


VIEWS: Tuple[str, ...] = ("top_cam", "side1_cam", "side2_cam")
DEFAULT_TAIL_SEED_OFFSET_PX = 8.0
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


def _prediction_slp_path(
    prediction_slp_root: Path,
    experiment: str,
    view: str,
    prediction_sleap_subdir: str | None,
) -> Path:
    base = prediction_slp_root / experiment
    if prediction_sleap_subdir:
        base = base / prediction_sleap_subdir
    return base / f"{view}.slp"


def _mouth_tail_instances_from_lf(
    src_lf,
    *,
    skeleton,
    tail_offset_xy: Tuple[float, float],
):
    import sleap_io as sio

    src_skeleton = src_lf.instances[0].skeleton if src_lf.instances else None
    src_nodes = [n.name for n in src_skeleton.nodes] if src_skeleton else []
    mouth_idx = src_nodes.index("mouth") if "mouth" in src_nodes else 0
    tail_idx = src_nodes.index("tail") if "tail" in src_nodes else None

    out = []
    for inst in src_lf.instances:
        pts = inst.numpy()
        if mouth_idx >= len(pts):
            continue
        mouth = pts[mouth_idx]
        if not np.isfinite(mouth).all():
            mask = np.isfinite(pts).all(axis=1)
            if not mask.any():
                continue
            mouth = pts[mask][0]
        mx, my = float(mouth[0]), float(mouth[1])

        tail = None
        if tail_idx is not None and tail_idx < len(pts):
            cand_tail = pts[tail_idx]
            if np.isfinite(cand_tail).all():
                tail = cand_tail

        if tail is not None:
            tx, ty = float(tail[0]), float(tail[1])
            tail_visible = True
        else:
            tx = mx + tail_offset_xy[0]
            ty = my + tail_offset_xy[1]
            tail_visible = False

        out.append(sio.Instance.from_numpy(
            points_data=np.array(
                [[mx, my, True], [tx, ty, tail_visible]],
                dtype=float,
            ),
            skeleton=skeleton,
        ))
    return out


def expand(
    seed_path: Path,
    out_path: Path,
    medaka_data_dir: Path,
    *,
    prediction_slp_root: Path | None = None,
    prediction_sleap_subdir: str | None = None,
    tail_offset_xy: Tuple[float, float] = (0.0, DEFAULT_TAIL_SEED_OFFSET_PX),
) -> dict:
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
    disk_video_cache: Dict[Tuple[str, str], "sio.Video"] = {}
    prediction_cache: Dict[Tuple[str, str], Dict[int, "sio.LabeledFrame"]] = {}
    prediction_skeleton = (
        src.skeletons[0]
        if src.skeletons
        else sio.Skeleton(nodes=["mouth", "tail"], edges=[("mouth", "tail")])
    )
    added = 0
    added_with_predictions = 0
    added_instances = 0
    no_video = 0
    no_prediction_slp = 0
    no_prediction_lf = 0
    pre_hist = defaultdict(int)
    for (exp, frame_idx), view_lfs in existing.items():
        pre_hist[len(view_lfs)] += 1
        missing = [v for v in VIEWS if v not in view_lfs]
        for view in missing:
            disk = medaka_data_dir / exp / view / VIEW_TO_VIDEO[view]
            video = None
            if disk.exists():
                cache_key = (exp, view)
                video = disk_video_cache.get(cache_key)
                if video is None:
                    try:
                        video = sio.load_video(str(disk))
                        disk_video_cache[cache_key] = video
                    except Exception:
                        video = None
            if video is None:
                # Fall back to an existing Video for this view. This may
                # point at the original drive recorded in the seed, so prefer
                # the explicit medaka_data_dir above when it is available.
                video = exp_view_to_video.get((exp, view))
            if video is None:
                no_video += 1
                continue
            instances = []
            if prediction_slp_root is not None:
                pred_key = (exp, view)
                if pred_key not in prediction_cache:
                    pred_slp = _prediction_slp_path(
                        prediction_slp_root, exp, view, prediction_sleap_subdir
                    )
                    if pred_slp.exists():
                        pred_labels = sio.load_slp(str(pred_slp))
                        prediction_cache[pred_key] = {
                            int(lf.frame_idx): lf
                            for lf in pred_labels.labeled_frames
                        }
                    else:
                        prediction_cache[pred_key] = {}
                        no_prediction_slp += 1
                pred_lf = prediction_cache[pred_key].get(int(frame_idx))
                if pred_lf is None:
                    no_prediction_lf += 1
                else:
                    instances = _mouth_tail_instances_from_lf(
                        pred_lf,
                        skeleton=prediction_skeleton,
                        tail_offset_xy=tail_offset_xy,
                    )
                    if instances:
                        added_with_predictions += 1
                        added_instances += len(instances)
            new_lfs.append(sio.LabeledFrame(
                video=video, frame_idx=int(frame_idx), instances=instances,
            ))
            added += 1

    print(f"  views-per-group before: {dict(pre_hist)}")
    print(f"  added {added} LF(s) for missing views (no_video={no_video}).")
    if prediction_slp_root is not None:
        print(
            "  filled missing views from predictions: "
            f"{added_with_predictions} LF(s), {added_instances} instance(s) "
            f"(no_prediction_slp={no_prediction_slp}, "
            f"no_prediction_lf={no_prediction_lf})."
        )

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
        "added_with_predictions": added_with_predictions,
        "added_instances": added_instances,
        "no_video": no_video,
        "no_prediction_slp": no_prediction_slp,
        "no_prediction_lf": no_prediction_lf,
        "out_path": str(out_path),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--paths-config", default=None)
    ap.add_argument("--medaka-data-dir", default=None)
    ap.add_argument(
        "--prediction-slp-root",
        default=None,
        help=(
            "Optional sleap_output root for filling added synchronized views "
            "with existing predictions, e.g. .../sleap_output."
        ),
    )
    ap.add_argument(
        "--prediction-sleap-subdir",
        default=None,
        help=(
            "Optional prediction subfolder inside each experiment, "
            "e.g. round4_260520."
        ),
    )
    ap.add_argument(
        "--tail-offset-px",
        type=float,
        default=DEFAULT_TAIL_SEED_OFFSET_PX,
        help=(
            "Hidden fallback tail offset used only when a prediction is "
            "missing its tail node."
        ),
    )
    args = ap.parse_args(argv)

    local = load_local_paths(args.paths_config)
    medaka_data_dir = choose_path(
        args.medaka_data_dir, local.medaka_data_dir if local else None,
    )
    if medaka_data_dir is None:
        ap.error("medaka_data_dir not set (paths.local.toml or --medaka-data-dir).")

    summary = expand(
        args.seed,
        args.out,
        Path(medaka_data_dir),
        prediction_slp_root=(
            Path(args.prediction_slp_root)
            if args.prediction_slp_root
            else None
        ),
        prediction_sleap_subdir=args.prediction_sleap_subdir,
        tail_offset_xy=(0.0, args.tail_offset_px),
    )
    print(summary)


if __name__ == "__main__":
    main()
