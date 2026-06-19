from __future__ import annotations

"""Measure whether the three camera views go ambiguous at the SAME time.

A concrete NN tracklet breaks on a view whenever two fish come within ~1.5 body
lengths *in that view's image*. But fish that overlap in the top (XY) view are
usually separated in depth -- which the side cameras see. So the views should
fail at *different* moments, and the other two can carry identity through one
view's gap. This script quantifies that:

  * per-view "clean" fraction (exactly num_fish detections, all separated >= sep),
  * the joint distribution of how many views are clean each frame (0/1/2/3),
  * and the key conditional: when the TOP view is NOT clean, how often is at least
    one (or both) side view clean -- i.e. how recoverable are top's gaps via 3D.

Per-view "all-3-separated" is a CONSERVATIVE proxy for 3D recoverability: real
cross-view triangulation can still resolve a frame where no single view has all
three apart, as long as the triplet matching is unambiguous. So treat ">=2 views
clean" as a lower bound on how much of the timeline 3D binding can anchor.

Run::

    python -m scripts.tracking.multiview_clean_coverage <exp> --paths-config configs/paths.local.toml
    python -m scripts.tracking.multiview_clean_coverage --slp-dir <dir>
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.common.local_paths import choose_path, load_local_paths
from scripts.tracking.concrete_tracklets import _load_frames_data, _min_pairwise

VIEWS = ["top_cam", "side1_cam", "side2_cam"]


def clean_frames_for_view(frames_data, *, num_fish, body_length, sep_bl, min_score):
    sep = float(sep_bl) * float(body_length)
    out = set()
    for f, dets in frames_data.items():
        if len(dets) != num_fish:
            continue
        if not all(s >= min_score for _, _, s in dets):
            continue
        pts = np.array([[x, y] for x, y, _ in dets], dtype=float)
        if _min_pairwise(pts) >= sep:
            out.add(f)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Multi-view clean-coverage / 3D-recoverability probe.")
    p.add_argument("experiment", nargs="?", default=None)
    p.add_argument("--slp-dir", default=None, help="folder holding <view>.slp (overrides experiment lookup)")
    p.add_argument("--num-fish", type=int, default=3)
    p.add_argument("--sep-bl", type=float, default=1.5)
    p.add_argument("--min-score", type=float, default=0.3)
    p.add_argument("--no-top-k", action="store_true")
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--end-frame", type=int, default=None)
    p.add_argument("--paths-config", default=None)
    p.add_argument("--sleap-output-dir", default=None)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    slp_dir = Path(args.slp_dir) if args.slp_dir else None
    if slp_dir is None:
        if args.experiment is None:
            raise SystemExit("pass an experiment name or --slp-dir")
        local = load_local_paths(args.paths_config)
        out_dir = choose_path(args.sleap_output_dir, local.sleap_output_dir if local else None)
        slp_dir = Path(out_dir) / args.experiment

    clean: dict[str, set] = {}
    present: dict[str, set] = {}
    for v in VIEWS:
        slp = slp_dir / f"{v}.slp"
        if not slp.exists():
            raise SystemExit(f"missing {slp}")
        print(f"loading {v} ...", flush=True)
        fd, bl, n_wrong = _load_frames_data(
            slp, num_fish=args.num_fish, top_k=not args.no_top_k,
            start_frame=args.start_frame, end_frame=args.end_frame,
        )
        cv = clean_frames_for_view(fd, num_fish=args.num_fish, body_length=bl,
                                   sep_bl=args.sep_bl, min_score=args.min_score)
        clean[v] = cv
        present[v] = set(fd)
        print(f"  {v}: frames={len(fd)}  body_length~{bl:.1f}px  "
              f"clean={len(cv)} ({100*len(cv)/max(len(fd),1):.1f}%)")

    # joint distribution over the union timeline
    all_frames = sorted(set().union(*present.values()))
    dist = Counter(sum(f in clean[v] for v in VIEWS) for f in all_frames)
    n = len(all_frames)
    print(f"\n=== joint clean coverage over {n} frames (union timeline) ===")
    for k in (3, 2, 1, 0):
        print(f"  exactly {k} view(s) clean: {dist[k]:>8d}  ({100*dist[k]/max(n,1):5.1f}%)")
    ge2 = dist[3] + dist[2]
    ge1 = ge2 + dist[1]
    print(f"  >=2 views clean (triangulable):   {ge2:>8d}  ({100*ge2/max(n,1):5.1f}%)")
    print(f"  >=1 view  clean:                  {ge1:>8d}  ({100*ge1/max(n,1):5.1f}%)")
    print(f"  ALL 3 views non-clean (hard):     {dist[0]:>8d}  ({100*dist[0]/max(n,1):5.1f}%)")

    # conditional: recoverability of the TOP view's gaps
    top_frames = present["top_cam"]
    not_clean_top = [f for f in top_frames if f not in clean["top_cam"]]
    m = len(not_clean_top)
    s1 = sum(f in clean["side1_cam"] for f in not_clean_top)
    s2 = sum(f in clean["side2_cam"] for f in not_clean_top)
    both = sum((f in clean["side1_cam"]) and (f in clean["side2_cam"]) for f in not_clean_top)
    either = sum((f in clean["side1_cam"]) or (f in clean["side2_cam"]) for f in not_clean_top)
    print(f"\n=== when TOP is NOT clean ({m} frames) -- can the sides resolve it? ===")
    print(f"  side1 clean: {100*s1/max(m,1):5.1f}%   side2 clean: {100*s2/max(m,1):5.1f}%")
    print(f"  >=1 side clean: {either:>8d} ({100*either/max(m,1):5.1f}%)   "
          f"both sides clean: {both:>8d} ({100*both/max(m,1):5.1f}%)")
    print(f"  neither side clean (top gap unrecoverable by sides alone): "
          f"{m-either} ({100*(m-either)/max(m,1):5.1f}%)")


if __name__ == "__main__":
    main()
