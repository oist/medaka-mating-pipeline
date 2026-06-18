from __future__ import annotations

"""
Split an annotator-bundle ``.pkg.slp`` into per-view training datasets and
generate matching sleap-nn training configs.

Workflow context
----------------
This is the standard HITL retraining step: after the colleague-bundled
PyQt annotator produces a multi-view ``annotation_seed.pkg.slp`` (one .slp
holding labels from top_cam + side1_cam + side2_cam interleaved), we need
two top-down model datasets:

    <round_dir>/top/labels.pkg.slp        # top_cam only
    <round_dir>/side/labels.pkg.slp       # side1_cam + side2_cam merged

The colleague-bundled .pkg.slp no longer carries the original mp4 paths on
the top-level ``Video.filename`` (those are rewritten to the .pkg.slp
itself when frames are embedded). The original path *is* preserved inside
the HDF5 group ``videoN/source_video`` as a JSON attribute — we walk that
JSON chain to recover the original mp4 filename and classify the view.

Outputs written per view (relative to ``--round-dir``):

    <view>/labels.pkg.slp                          # split labels, embed='user'
    <view>/configs/centroid.yaml                   # sleap-nn config (disk cache)
    <view>/configs/centered_instance.yaml          # sleap-nn config (disk cache)

If a previous ``labels.pkg.slp`` exists at the destination it is renamed
to ``labels_<backup_tag>.pkg.slp`` before being overwritten so prior
round artefacts are not lost.

Usage
-----
::

    python -m scripts.sleap.split_pkg_by_view \
        --seed C:/Users/machi/Downloads/annotator_bundle_uv/data/annotation_seed.pkg.slp \
        --round-dir Z:/StephensU/medaka/sleap_models/retrain_20260508 \
        --round-tag round4_260520 \
        --cache-root C:/sleap_cache \
        --backup-tag round3
"""

import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import h5py
import sleap_io as sio
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_DIR = REPO_ROOT / "configs" / "sleap" / "training"

VIEW_TAGS = ("top_cam", "side1_cam", "side2_cam")
VIEW_TO_BUCKET = {
    "top_cam": "top",
    "side1_cam": "side",
    "side2_cam": "side",
}
BUCKETS = ("top", "side")
KINDS = ("centroid", "centered_instance")


# ---------------------------------------------------------------------------
# View classification
# ---------------------------------------------------------------------------

def _walk_source_filename(json_blob: dict) -> str | None:
    """Walk a videoN/source_video JSON chain to the leaf recording path.

    The JSON nests ``source_video`` inside itself, one layer per embedding
    round-trip: HDF5Video (this .pkg.slp) -> HDF5Video (annotation seed) ->
    ... -> MediaVideo (the original mp4). We descend through every embedded
    wrapper and return the first path that is clearly the source recording:
    a ``MediaVideo`` backend, or any filename already carrying a view tag.

    Earlier this short-circuited on "first filename not ending in .pkg.slp",
    which broke on newer annotator bundles that wrap the data in an
    ``edits.pkg_full.slp`` (ends in ``.slp``, not ``.pkg.slp``) — that guard
    returned the wrapper path (no view tag) instead of descending to the mp4.
    """
    cur = json_blob
    last_fname: str | None = None
    for _ in range(12):
        if not isinstance(cur, dict):
            break
        backend = cur.get("backend", {}) or {}
        fname = backend.get("filename") or cur.get("filename")
        btype = backend.get("type")
        if fname:
            last_fname = fname
        if btype == "MediaVideo" and fname:
            return fname
        if fname and _classify_path(fname) is not None:
            return fname
        nxt = cur.get("source_video")
        if nxt is None:
            break
        cur = nxt
    return last_fname


def _classify_path(path: str | None) -> str | None:
    if not path:
        return None
    s = str(path).replace("\\", "/").lower()
    for tag in VIEW_TAGS:
        if tag in s:
            return tag
    return None


def classify_videos(seed_path: Path) -> dict[int, str | None]:
    """Return ``{video_index: view_tag or None}`` for every video group."""
    out: dict[int, str | None] = {}
    with h5py.File(str(seed_path), "r") as f:
        for key in f.keys():
            if not key.startswith("video"):
                continue
            try:
                idx = int(key[len("video"):])
            except ValueError:
                continue
            sv = f.get(f"{key}/source_video")
            view: str | None = None
            if sv is not None:
                raw = sv.attrs.get("json")
                if isinstance(raw, (bytes, bytearray)):
                    raw = raw.decode("utf-8", errors="replace")
                if isinstance(raw, str):
                    try:
                        blob = json.loads(raw)
                    except json.JSONDecodeError:
                        blob = None
                    if blob is not None:
                        leaf = _walk_source_filename(blob)
                        view = _classify_path(leaf)
            out[idx] = view
    return out


# ---------------------------------------------------------------------------
# Splitting labels
# ---------------------------------------------------------------------------

def _video_dataset_index(video: "sio.Video") -> int | None:
    """Return the videoN HDF5 group index for a sleap-io Video.

    The Video keeps the dataset path (``videoN/video``) on its backend; we
    parse N from it. Falls back to ``source_video.backend.dataset`` since
    a re-saved pkg may wrap the embedded video twice.
    """
    cur = video
    for _ in range(4):
        if cur is None:
            return None
        backend = getattr(cur, "backend", None)
        ds = getattr(backend, "dataset", None) if backend is not None else None
        if isinstance(ds, str) and ds.startswith("video"):
            n = ds[len("video"):].split("/", 1)[0]
            if n.isdigit():
                return int(n)
        cur = getattr(cur, "source_video", None)
    return None


def split_labels(
    seed_path: Path,
    bucket_dirs: dict[str, Path],
) -> dict[str, dict]:
    """Split ``seed_path`` into per-bucket pkg.slp files. Returns stats."""
    labels = sio.load_slp(str(seed_path))
    view_for_dataset = classify_videos(seed_path)

    # Build bucket -> list of Videos (preserving insertion order).
    bucket_videos: dict[str, list] = defaultdict(list)
    seen_per_bucket: dict[str, set[int]] = defaultdict(set)
    unknown = 0
    view_hist: Counter[str] = Counter()
    for v in labels.videos:
        ds_idx = _video_dataset_index(v)
        view = view_for_dataset.get(ds_idx) if ds_idx is not None else None
        view_hist[view or "unknown"] += 1
        bucket = VIEW_TO_BUCKET.get(view) if view else None
        if bucket is None:
            unknown += 1
            continue
        if id(v) in seen_per_bucket[bucket]:
            continue
        seen_per_bucket[bucket].add(id(v))
        bucket_videos[bucket].append(v)

    print(f"  videos by view: {dict(view_hist)}")
    if unknown:
        print(f"  WARNING: {unknown} video(s) had no recognisable view tag; skipped.")

    # Build bucket -> list of LabeledFrames (non-empty only).
    bucket_lfs: dict[str, list] = defaultdict(list)
    skipped_empty = 0
    lf_view_hist: Counter[str] = Counter()
    for lf in labels.labeled_frames:
        ds_idx = _video_dataset_index(lf.video)
        view = view_for_dataset.get(ds_idx) if ds_idx is not None else None
        lf_view_hist[view or "unknown"] += 1
        bucket = VIEW_TO_BUCKET.get(view) if view else None
        if bucket is None:
            continue
        if not lf.instances:
            skipped_empty += 1
            continue
        bucket_lfs[bucket].append(lf)
    print(f"  LFs by view (all): {dict(lf_view_hist)}")
    if skipped_empty:
        print(f"  dropped {skipped_empty} empty LF(s) (no user instances).")

    stats: dict[str, dict] = {}
    for bucket in BUCKETS:
        out_path = bucket_dirs[bucket] / "labels.pkg.slp"
        vids = bucket_videos.get(bucket, [])
        lfs = bucket_lfs.get(bucket, [])
        n_inst = sum(len(lf.instances) for lf in lfs)
        if not lfs:
            print(f"  [{bucket}] no LFs; skipping write.")
            stats[bucket] = {"n_lfs": 0, "n_videos": 0, "n_instances": 0,
                              "out_path": str(out_path)}
            continue
        # Backup existing labels.pkg.slp if present (and no backup yet).
        if out_path.exists():
            backup = bucket_dirs[bucket] / f"labels_{_BACKUP_TAG}.pkg.slp"
            if not backup.exists():
                shutil.move(str(out_path), str(backup))
                print(f"  [{bucket}] backed up old labels -> {backup.name}")
            else:
                print(f"  [{bucket}] {backup.name} exists; overwriting in place.")
                out_path.unlink()
        suggestions = [
            sio.SuggestionFrame(video=lf.video, frame_idx=int(lf.frame_idx))
            for lf in lfs
        ]
        out_labels = sio.Labels(
            skeletons=labels.skeletons,
            videos=vids,
            labeled_frames=lfs,
            suggestions=suggestions,
            provenance={"source_seed": str(seed_path), "bucket": bucket},
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  [{bucket}] saving {len(lfs)} LFs / {n_inst} instances / "
              f"{len(vids)} videos -> {out_path}")
        sio.save_slp(out_labels, str(out_path), embed="user")
        size_mb = out_path.stat().st_size / 1e6
        print(f"  [{bucket}]   wrote {size_mb:.1f} MB")
        stats[bucket] = {
            "n_lfs": len(lfs), "n_videos": len(vids), "n_instances": n_inst,
            "out_path": str(out_path),
        }
    return stats


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------

def _load_template(kind: str) -> dict:
    p = TEMPLATE_DIR / f"{kind}.yaml"
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_configs(
    bucket: str,
    bucket_dir: Path,
    *,
    labels_path: Path,
    runs_dir: Path,
    cache_root: Path,
    round_tag: str,
    n_labels: int,
) -> list[Path]:
    """Write per-round ``centroid.yaml`` and ``centered_instance.yaml``."""
    out_paths: list[Path] = []
    configs_dir = bucket_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    for kind in KINDS:
        cfg = _load_template(kind)
        cfg["data_config"]["train_labels_path"] = [str(labels_path)]
        # Disk cache for BOTH centroid and centered_instance (user request).
        cfg["data_config"]["data_pipeline_fw"] = "torch_dataset_cache_img_disk"
        cfg["data_config"]["cache_img_path"] = str(
            cache_root / f"{bucket_dir.parent.name}_{round_tag}_{bucket}_{kind}"
        )
        cfg["trainer_config"]["ckpt_dir"] = str(runs_dir)
        cfg["trainer_config"]["run_name"] = f"{round_tag}.{kind}_n{n_labels}"
        out = configs_dir / f"{kind}.yaml"
        with out.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        out_paths.append(out)
        print(f"  [{bucket}/{kind}] wrote {out}")
    return out_paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_BACKUP_TAG = "prevround"  # overwritten in main() from --backup-tag


def main(argv: Iterable[str] | None = None) -> int:
    global _BACKUP_TAG
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=Path, required=True,
                    help="Annotator-bundle pkg.slp (multi-view labels).")
    ap.add_argument("--round-dir", type=Path, required=True,
                    help="Destination round directory, e.g. "
                         "Z:/StephensU/medaka/sleap_models/retrain_20260508.")
    ap.add_argument("--round-tag", required=True,
                    help="Run-name tag injected into configs (e.g. round4_260520).")
    ap.add_argument("--cache-root", type=Path,
                    default=Path("C:/sleap_cache"),
                    help="Root for sleap-nn disk cache (one subdir per view+kind).")
    ap.add_argument("--backup-tag", default="round3",
                    help="Suffix for backing up the existing labels.pkg.slp "
                         "(only used if no backup already exists). Default: round3.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Classify and report, but don't write any files.")
    args = ap.parse_args(list(argv) if argv is not None else None)

    if not args.seed.exists():
        ap.error(f"seed not found: {args.seed}")
    if not args.round_dir.exists():
        ap.error(f"round-dir not found: {args.round_dir}")
    _BACKUP_TAG = args.backup_tag

    print(f"seed       : {args.seed}")
    print(f"round-dir  : {args.round_dir}")
    print(f"round-tag  : {args.round_tag}")
    print(f"cache-root : {args.cache_root}")
    print()

    if args.dry_run:
        labels = sio.load_slp(str(args.seed))
        view_for_dataset = classify_videos(args.seed)
        per_bucket_lfs: Counter[str] = Counter()
        per_bucket_inst: Counter[str] = Counter()
        for lf in labels.labeled_frames:
            ds_idx = _video_dataset_index(lf.video)
            view = view_for_dataset.get(ds_idx) if ds_idx is not None else None
            bucket = VIEW_TO_BUCKET.get(view) if view else "unknown"
            if not lf.instances:
                continue
            per_bucket_lfs[bucket] += 1
            per_bucket_inst[bucket] += len(lf.instances)
        print("dry-run summary:")
        for b in BUCKETS + ("unknown",):
            print(f"  {b:<8} LFs={per_bucket_lfs[b]}  instances={per_bucket_inst[b]}")
        return 0

    bucket_dirs = {b: args.round_dir / b for b in BUCKETS}
    for b, d in bucket_dirs.items():
        d.mkdir(parents=True, exist_ok=True)

    stats = split_labels(args.seed, bucket_dirs)
    print()
    print("Writing per-round configs (disk cache; both centroid and centered_instance)...")
    for bucket in BUCKETS:
        if stats[bucket]["n_lfs"] == 0:
            print(f"  [{bucket}] skipped (no labels).")
            continue
        write_configs(
            bucket,
            bucket_dirs[bucket],
            labels_path=bucket_dirs[bucket] / "labels.pkg.slp",
            runs_dir=bucket_dirs[bucket] / "runs",
            cache_root=args.cache_root,
            round_tag=args.round_tag,
            n_labels=stats[bucket]["n_lfs"],
        )

    print()
    print("Done. Launch training with:")
    print(
        f'  .venv\\Scripts\\python.exe -m scripts.sleap.run_round3_training '
        f'--round-dir "{args.round_dir}" --round-tag {args.round_tag}'
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
