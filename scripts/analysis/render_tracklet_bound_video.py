"""Overlay the per-view tracklet-bound identity (bind_tracklets_3d output) on the
three camera videos for a window, so a human can watch the stitched identity track
each fish.

Each fish's bound identity (global_id) gets a fixed color; a filled dot + id is drawn
at the detection on every view where that fish has a tracklet this frame. Frames the
binder flagged `in_hardcore` (a genuine 3D contact / decision point) draw the marker
hollow, so you can see tracking hold through slow approaches and only go uncertain at
real crossings. The banner shows how many of the 3 fish are tracked per view.

Run::

    python -m scripts.analysis.render_tracklet_bound_video \
        --bind-dir C:/udmt_validation/tracklet_bind_stifd \
        --video-top ... --video-side1 ... --video-side2 ... \
        --start-frame 170000 --end-frame 171000 --output out.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from scripts.analysis.render_predictions_identity_video import ID_COLORS, TARGET_H

VIEWS = [("top", "top_cam"), ("side1", "side1_cam"), ("side2", "side2_cam")]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bind-dir", type=Path, required=True)
    ap.add_argument("--video-top", type=Path, required=True)
    ap.add_argument("--video-side1", type=Path, required=True)
    ap.add_argument("--video-side2", type=Path, required=True)
    ap.add_argument("--start-frame", type=int, required=True)
    ap.add_argument("--end-frame", type=int, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--id-col", default="global_id")
    ap.add_argument("--fps", type=float, default=166.0)
    ap.add_argument("--target-h", type=int, default=TARGET_H)
    args = ap.parse_args()

    videos = {"top": args.video_top, "side1": args.video_side1, "side2": args.video_side2}
    dfs = {}
    ids = set()
    for short, cam in VIEWS:
        csv = args.bind_dir / f"{cam}.tracklet_bound.csv"
        if not csv.exists():
            raise SystemExit(f"missing {csv}")
        d = pd.read_csv(csv)
        d = d[(d["frame"] >= args.start_frame) & (d["frame"] <= args.end_frame)]
        dfs[short] = {f: sub for f, sub in d.groupby("frame")}
        ids.update(int(v) for v in d[args.id_col].dropna().unique())
    cidx = {fid: i for i, fid in enumerate(sorted(ids))}   # stable color per fish

    caps = {s: cv2.VideoCapture(str(videos[s])) for s, _ in VIEWS}
    for s, c in caps.items():
        if not c.isOpened():
            raise SystemExit(f"cannot open {videos[s]}")
    w = int(caps["top"].get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(caps["top"].get(cv2.CAP_PROP_FRAME_HEIGHT))
    th = args.target_h
    nw = int(w * th / h)
    out_w, out_h = nw * 3, th + 34
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (out_w, out_h))
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **k):
            return x

    read_pos = {s: -10 for s, _ in VIEWS}
    for fi in tqdm(range(args.start_frame, args.end_frame + 1), desc="render", ncols=80):
        panels, counts = [], []
        for short, _cam in VIEWS:
            cap = caps[short]
            if fi != read_pos[short] + 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            read_pos[short] = fi
            if not ret or frame is None:
                frame = np.zeros((h, w, 3), np.uint8)
            sub = dfs[short].get(fi)
            n = 0
            if sub is not None:
                for r in sub.itertuples():
                    fid = int(getattr(r, args.id_col))
                    col = ID_COLORS[cidx.get(fid, 0) % len(ID_COLORS)]
                    x, y = int(r.x), int(r.y)
                    hard = bool(getattr(r, "in_hardcore", False))
                    cv2.circle(frame, (x, y), 11, col, 2 if hard else -1)  # hollow in hard-core
                    cv2.putText(frame, str(fid), (x + 13, y + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv2.LINE_AA)
                    n += 1
            cv2.putText(frame, short, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            panels.append(cv2.resize(frame, (nw, th)))
            counts.append(n)
        bar = np.zeros((34, out_w, 3), np.uint8)
        cv2.putText(bar, f"t={fi/args.fps:7.2f}s  frame {fi}   fish tracked  "
                    f"top:{counts[0]} side1:{counts[1]} side2:{counts[2]}   "
                    f"(hollow = in hard-core)", (10, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        writer.write(np.vstack([np.hstack(panels), bar]))

    writer.release()
    for c in caps.values():
        c.release()
    print(f"Wrote: {args.output}  ({args.end_frame - args.start_frame + 1} frames, "
          f"{(args.end_frame - args.start_frame + 1)/args.fps:.0f}s)")


if __name__ == "__main__":
    main()
