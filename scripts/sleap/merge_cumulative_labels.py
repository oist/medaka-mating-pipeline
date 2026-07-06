from __future__ import annotations

"""
Merge cumulative per-view SLEAP training labels across rounds.

Motivation
----------
Round-5 (``retrain_20260611``) was trained on K_1_2 only (top n=900, side
n=1800). The round-6b annotation bundle adds two new experiments (K_1_1,
K_1_3) plus difficult close-contact K_1_2 frames, but it is NOT cumulative on
its own (top n=395, side n=790). To "train on all annotation round 1..6" we
must union the round-5 cumulative per-view labels with the round-6b bundle,
de-duplicating any frame that appears in both.

Dedup key
---------
The same physical recording appears under two path spellings on this machine
(``//bucket.oist.jp/bucket/.../medaka_202512/<EXP>/<view>/camN.mp4`` and
``Z:/StephensU/medaka/medaka_202512/<EXP>/<view>/camN.mp4``). We normalise
both to the tail after ``medaka_202512/`` and dedup on
``(normalised_clip, frame_idx)``. On a collision the round-6b annotation WINS
(it is the newer, difficulty-focused correction).

Inputs
------
- ``--round5-dir``: a round dir already split by view, each holding
  ``<view>/labels.pkg.slp`` (top, side).
- ``--round6b-seed``: the annotator-bundle ``edits.pkg_full.slp`` holding
  top_cam + side1_cam + side2_cam interleaved (embedded frames).

Output
------
``<out-dir>/<view>/labels.pkg.slp`` (embed="user") for view in {top, side},
plus a ``merge_report.json`` summarising per-source / per-experiment counts and
any dropped degenerate instances.
"""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
import sleap_io as sio

# Reuse the battle-tested view classifier / source-walk from split_pkg_by_view.
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from split_pkg_by_view import (  # noqa: E402
    VIEW_TO_BUCKET,
    _video_dataset_index,
    _walk_source_filename,
)

BUCKETS = ("top", "side")
_MARKER = "medaka_202512/"


def _norm_clip(path: str | None) -> str | None:
    """Normalise a recording path to ``<EXP>/<view>/<file>`` (drive-agnostic)."""
    if not path:
        return None
    s = str(path).replace("\\", "/").lower()
    i = s.find(_MARKER)
    if i >= 0:
        return s[i + len(_MARKER):]
    # Marker absent (non-standard path): key on the FULL normalised path. Do NOT
    # strip to the view tag — that drops the experiment folder, so two different
    # experiments' same-view clips would collide and one would be false-deduped.
    return s


def _leaf_map(pkg_path: Path) -> dict[int, str | None]:
    """``video_dataset_index -> leaf source mp4 path`` for an embedded pkg."""
    out: dict[int, str | None] = {}
    with h5py.File(str(pkg_path), "r") as f:
        for key in f.keys():
            if not key.startswith("video"):
                continue
            try:
                idx = int(key[len("video"):])
            except ValueError:
                continue
            sv = f.get(f"{key}/source_video")
            leaf: str | None = None
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
            out[idx] = leaf
    return out


def _bucket_for_clip(norm_clip: str | None) -> str | None:
    if not norm_clip:
        return None
    for tag, bucket in (("top_cam", "top"), ("side1_cam", "side"),
                        ("side2_cam", "side")):
        if tag in norm_clip:
            return bucket
    return None


def _experiment_of(norm_clip: str | None) -> str:
    if not norm_clip:
        return "unknown"
    return norm_clip.split("/", 1)[0]


def _instance_is_degenerate(inst, mi: int, ti: int, max_len: float) -> bool:
    """Flag mouth/tail instances whose span is impossibly long (bad label)."""
    pts = inst.numpy()
    m, t = pts[mi], pts[ti]
    if not (np.all(np.isfinite(m)) and np.all(np.isfinite(t))):
        return False  # partial labels are legitimate; keep them
    return bool(math.hypot(*(m - t)) > max_len)


def _collect(pkg_path: Path, priority: bool, max_len: float):
    """Return {bucket: {(norm_clip, frame_idx): (LabeledFrame, priority)}}.

    Also returns provenance counters. ``priority`` marks round-6b so it wins
    collisions.
    """
    labels = sio.load_slp(str(pkg_path))
    leaf_map = _leaf_map(pkg_path)
    skel = labels.skeletons[0]
    names = [n.name for n in skel.nodes]
    mi, ti = names.index("mouth"), names.index("tail")

    per_bucket: dict[str, dict[tuple, object]] = {b: {} for b in BUCKETS}
    exp_counter: Counter = Counter()
    dropped_degenerate = 0
    skipped_no_view = 0
    for lf in labels.labeled_frames:
        if not lf.instances:
            continue
        ds = _video_dataset_index(lf.video)
        leaf = leaf_map.get(ds) if ds is not None else None
        norm = _norm_clip(leaf)
        bucket = _bucket_for_clip(norm)
        if bucket is None:
            skipped_no_view += 1
            continue
        # Drop degenerate instances (e.g. the 1419 px side outlier) in place.
        kept = [i for i in lf.instances
                if not _instance_is_degenerate(i, mi, ti, max_len)]
        dropped_degenerate += len(lf.instances) - len(kept)
        if not kept:
            continue
        lf.instances = kept
        key = (norm, int(lf.frame_idx))
        per_bucket[bucket][key] = (lf, priority)
        exp_counter[(bucket, _experiment_of(norm))] += 1
    return (labels, per_bucket, exp_counter, dropped_degenerate,
            skipped_no_view, skel)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--round5-dir", type=Path, required=True,
                    help="Round dir with <view>/labels.pkg.slp (cumulative 1..5).")
    ap.add_argument("--round6b-seed", type=Path, required=True,
                    help="Round-6b annotator bundle edits.pkg_full.slp.")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Output round dir; writes <view>/labels.pkg.slp.")
    ap.add_argument("--max-body-len", type=float, default=250.0,
                    help="Drop mouth->tail instances longer than this (px). "
                         "Real medaka max ~165 px; guards against mislabels.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    r5_top = args.round5_dir / "top" / "labels.pkg.slp"
    r5_side = args.round5_dir / "side" / "labels.pkg.slp"
    for p in (r5_top, r5_side, args.round6b_seed):
        if not p.exists():
            ap.error(f"input not found: {p}")

    print(f"[1/3] Reading round-6b seed (priority): {args.round6b_seed}")
    (r6_labels, r6_bucket, r6_exp, r6_drop, r6_skip, r6_skel) = _collect(
        args.round6b_seed, priority=True, max_len=args.max_body_len)
    r6_names = [n.name for n in r6_skel.nodes]
    total_drop = r6_drop
    total_skip = r6_skip
    print(f"   round-6b by bucket: "
          f"{ {b: len(r6_bucket[b]) for b in BUCKETS} }  "
          f"(dropped {r6_drop} degenerate, {r6_skip} no-view)")

    # Round-5 is two separate per-view pkgs.
    print(f"[2/3] Reading round-5 cumulative: {args.round5_dir}")
    merged: dict[str, dict[tuple, object]] = {b: dict(r6_bucket[b]) for b in BUCKETS}
    exp_counter: Counter = Counter(r6_exp)
    collisions: Counter = Counter()
    for view_pkg, want_bucket in ((r5_top, "top"), (r5_side, "side")):
        (_, r5_bucket, r5_exp, r5_drop, r5_skip, r5_skel) = _collect(
            view_pkg, priority=False, max_len=args.max_body_len)
        # Sources are loaded separately, so their Skeleton objects differ by
        # identity even when equal. sleap-nn keys confmap channels by node ORDER
        # (inst.numpy()), so a reversed skeleton would silently swap mouth<->tail.
        # Assert identical node order before we remap everything onto r6_skel.
        r5_names = [n.name for n in r5_skel.nodes]
        if r5_names != r6_names:
            raise SystemExit(
                f"skeleton node-order mismatch: {view_pkg} has {r5_names} "
                f"but round-6b has {r6_names}; refusing to merge.")
        total_drop += r5_drop
        total_skip += r5_skip
        for b in BUCKETS:
            for key, val in r5_bucket[b].items():
                if key in merged[b]:
                    collisions[b] += 1  # round-6b already holds it; skip round-5
                    continue
                merged[b][key] = val
        for k, v in r5_exp.items():
            exp_counter[k] += v
        print(f"   {view_pkg.parent.name}: bucket sizes "
              f"{ {b: len(r5_bucket[b]) for b in BUCKETS} } "
              f"(dropped {r5_drop} degenerate, {r5_skip} no-view)")

    print(f"   collisions (round-6b won): {dict(collisions)}")
    print("   merged frames by bucket:",
          {b: len(merged[b]) for b in BUCKETS})
    print("   frames by (bucket, experiment):")
    for (b, exp), n in sorted(exp_counter.items()):
        print(f"      {b:<5} {exp:<24} {n}")

    report = {
        "round5_dir": str(args.round5_dir),
        "round6b_seed": str(args.round6b_seed),
        "out_dir": str(out_dir),
        "max_body_len": args.max_body_len,
        "merged_by_bucket": {b: len(merged[b]) for b in BUCKETS},
        "dropped_degenerate": total_drop,
        "skipped_no_view": total_skip,
        "collisions_round6b_won": dict(collisions),
        "by_bucket_experiment": {f"{b}|{exp}": n
                                 for (b, exp), n in exp_counter.items()},
    }

    if args.dry_run:
        print("\n[3/3] --dry-run: not writing labels.")
        print(json.dumps(report, indent=2))
        return 0

    print(f"[3/3] Writing merged per-view labels under {out_dir}")
    for b in BUCKETS:
        entries = list(merged[b].values())
        lfs = [lf for (lf, _prio) in entries]
        if not lfs:
            print(f"   [{b}] no frames; skipping.")
            continue
        # Remap every instance onto the single canonical skeleton (node order
        # already asserted equal) so the written pkg is genuinely single-skeleton
        # rather than carrying one identical Skeleton object per source load.
        for lf in lfs:
            for inst in lf.instances:
                inst.skeleton = r6_skel
        used_videos = list({id(lf.video): lf.video for lf in lfs}.values())
        suggestions = [sio.SuggestionFrame(video=lf.video,
                                           frame_idx=int(lf.frame_idx))
                       for lf in lfs]
        out_labels = sio.Labels(
            skeletons=[r6_skel],
            videos=used_videos,
            labeled_frames=lfs,
            suggestions=suggestions,
            provenance={"merge": "round1..6",
                        "round5_dir": str(args.round5_dir),
                        "round6b_seed": str(args.round6b_seed)},
        )
        out_path = out_dir / b / "labels.pkg.slp"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        n_inst = sum(len(lf.instances) for lf in lfs)
        print(f"   [{b}] {len(lfs)} LFs / {n_inst} instances / "
              f"{len(used_videos)} videos -> {out_path}")
        sio.save_slp(out_labels, str(out_path), embed="user")
        print(f"   [{b}]   wrote {out_path.stat().st_size / 1e6:.1f} MB")
        report[f"{b}_out"] = {"n_lfs": len(lfs), "n_instances": n_inst,
                              "n_videos": len(used_videos),
                              "path": str(out_path)}

    report_path = out_dir / "merge_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
