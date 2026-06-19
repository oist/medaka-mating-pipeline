"""Render a montage of the tightest fish-fish crossings for ID-quality checking.

ID swaps happen at occlusions and (because the 3D tracks stay geometrically
smooth through a swap) leave no pure-data signature — they must be checked by eye.
This finds the N closest 3D approaches between fish, and renders a back-to-back
montage of short windows around each, so you watch only the risky moments instead
of the full 35-min recording.

Each montage frame shows all three views with the REAL SLEAP mouth/tail
predictions colored by unified 3D identity (mouth=dot, tail=ring), the 3D
reprojection dots below, and a banner with the event number, time, and the
minimum inter-fish distance. The three IDs get distinct fixed colors — identify
the female by eye (side view) once, then check her color stays on her through
each crossing.

Usage::

    python -m scripts.analysis.render_crossing_montage \
        --csv-3d <...>/round5_260611_3d_identified.csv \
        --slp-dir <...>/sleap_output/round5_260611/<exp> \
        --model-folder calibration_results/.../models \
        --video-top ... --video-side1 ... --video-side2 ... \
        --output <...>/round5_260611_crossing_montage.mp4 \
        --n-events 40 --window-s 1.5 --close-mm 18
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from scripts.reconstruction_3d.calibration import Calibration
from scripts.analysis.render_predictions_identity_video import (
    load_view_predictions, draw_real, draw_reproj,
    VIEWS, VIEW_CAM_IDX, ID_COLORS, TARGET_H,
)

FPS = 166.0


def rank_crossings(df, id_col, close_mm, n_events, min_sep):
    """Return up to n_events (center_frame, min_mm) for the tightest approaches."""
    piv = df.pivot_table(index="frame", columns=id_col, values=["X", "Y", "Z"])
    frames = piv.index.to_numpy()
    ids = sorted(int(c) for c in df[id_col].dropna().unique())
    pts = {i: np.stack([piv[("X", i)], piv[("Y", i)], piv[("Z", i)]], axis=1) for i in ids}
    # min pairwise distance per frame
    mind = np.full(len(frames), np.inf)
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            d = np.linalg.norm(pts[ids[a]] - pts[ids[b]], axis=1)
            mind = np.minimum(mind, np.nan_to_num(d, nan=np.inf))
    # candidate frames below threshold, sorted by tightness; greedily pick spaced peaks
    order = np.argsort(mind)
    chosen = []
    for k in order:
        if mind[k] >= close_mm or not np.isfinite(mind[k]):
            break
        f = int(frames[k])
        if all(abs(f - c) >= min_sep for c, _ in chosen):
            chosen.append((f, float(mind[k])))
        if len(chosen) >= n_events:
            break
    chosen.sort()  # chronological for the montage
    return chosen, ids


def reproj_for_frames(df, cal, id_col, frame_set, views=VIEWS):
    sub = df[df["frame"].isin(frame_set)].dropna(subset=["X", "Y", "Z"])
    uv = cal.compute_imCoords_from_XYZ(sub[["X", "Y", "Z"]].to_numpy(float))
    out = defaultdict(lambda: {v: [] for v in views})
    fr = sub["frame"].to_numpy()
    fid = sub[id_col].to_numpy()
    for i in range(len(sub)):
        if not np.isfinite(fid[i]):
            continue
        f = int(fr[i])
        for v in views:
            x, y = uv[i, VIEW_CAM_IDX[v], 0], uv[i, VIEW_CAM_IDX[v], 1]
            if np.isfinite(x) and np.isfinite(y):
                out[f][v].append((int(fid[i]), x, y))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv-3d", type=Path, required=True)
    ap.add_argument("--slp-dir", type=Path, required=True)
    ap.add_argument("--model-folder", type=Path, required=True)
    ap.add_argument("--video-top", type=Path, default=None)
    ap.add_argument("--video-side1", type=Path, default=None)
    ap.add_argument("--video-side2", type=Path, default=None)
    ap.add_argument("--views", default=",".join(VIEWS),
                    help="Comma-separated subset of views to render (default all: "
                         f"{','.join(VIEWS)}). Use e.g. 'side1,side2' to skip an "
                         "unreadable top video; only the selected views' --video-* "
                         "are required and the montage width shrinks to fit.")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=FPS)
    ap.add_argument("--n-events", type=int, default=40)
    ap.add_argument("--events-csv", type=Path, default=None,
                    help="Render THESE events instead of auto-ranked crossings. Needs a "
                         "'center' column or 'start'+'end' (center = midpoint); 'min_mm' "
                         "optional for the banner. E.g. the resolver's hardcore_events_resolved.csv.")
    ap.add_argument("--events-filter", default=None,
                    help="pandas query() on --events-csv, e.g. \"verdict=='clear'\".")
    ap.add_argument("--events-sort", default=None,
                    help="sort --events-csv ascending by this column before taking n-events "
                         "(e.g. min_mm -> closest-approach / riskiest first).")
    ap.add_argument("--window-s", type=float, default=1.5)
    ap.add_argument("--close-mm", type=float, default=18.0)
    ap.add_argument("--assign-max-dist", type=float, default=60.0)
    ap.add_argument("--target-h", type=int, default=TARGET_H,
                    help="Per-view panel height in px; raise for higher resolution "
                         "(native frame is square ~1200px).")
    ap.add_argument("--no-reproj-row", action="store_true",
                    help="Drop the 3D-reprojection dot row so all the resolution "
                         "goes to the video panels (what you read the fish from).")
    ap.add_argument("--zoom-crop", type=int, default=0,
                    help="If >0, crop each view to a square of this many px centered "
                         "on the fish (then upscale to the panel) so the fish fills "
                         "the frame — best for reading body shape. E.g. 500.")
    args = ap.parse_args()

    sel_views = [v.strip() for v in args.views.split(",") if v.strip()]
    bad = [v for v in sel_views if v not in VIEWS]
    if bad:
        raise SystemExit(f"unknown view(s) {bad}; choose from {VIEWS}")
    videos = {"top": args.video_top, "side1": args.video_side1, "side2": args.video_side2}
    missing = [v for v in sel_views if videos[v] is None]
    if missing:
        raise SystemExit(f"--video-* required for selected view(s): {missing}")

    df = pd.read_csv(args.csv_3d, usecols=lambda c: c in
                     ("frame", "final_fish_id", "fish_id", "X", "Y", "Z"))
    id_col = "final_fish_id" if "final_fish_id" in df.columns else "fish_id"
    win = int(args.window_s * args.fps)

    if args.events_csv:
        ev = pd.read_csv(args.events_csv)
        if args.events_filter:
            ev = ev.query(args.events_filter)
        if "center" not in ev.columns:
            ev = ev.assign(center=((ev["start"] + ev["end"]) // 2).astype(int))
        if args.events_sort and args.events_sort in ev.columns:
            ev = ev.sort_values(args.events_sort)
        ev = ev.head(args.n_events)
        events = [(int(r.center),
                   float(r.min_mm) if "min_mm" in ev.columns and np.isfinite(r.min_mm) else np.nan)
                  for r in ev.itertuples()]
        events.sort()  # chronological for playback
        ids = sorted(int(c) for c in df[id_col].dropna().unique())
        if not events:
            raise SystemExit("--events-csv produced no events after filter")
    else:
        events, ids = rank_crossings(df, id_col, args.close_mm, args.n_events, min_sep=win)
        if not events:
            raise SystemExit("no crossings below threshold")
    id_to_cidx = {fid: i for i, fid in enumerate(ids)}  # sorted -> distinct colors

    # montage frame list (chronological, de-duped) + per-frame event meta
    frame_meta = {}
    for ev_i, (center, mind) in enumerate(events):
        for f in range(center - win, center + win + 1):
            if f < 0:
                continue
            if f not in frame_meta:
                frame_meta[f] = (ev_i + 1, mind, center)
    montage_frames = sorted(frame_meta)

    cal = Calibration(args.model_folder)
    reproj = reproj_for_frames(df, cal, id_col, set(montage_frames), sel_views)
    preds = {v: load_view_predictions(args.slp_dir / f"{ 'top_cam' if v=='top' else v+'_cam' }.slp")
             for v in sel_views}

    caps = {v: cv2.VideoCapture(str(videos[v])) for v in sel_views}
    for v, c in caps.items():
        if not c.isOpened():
            raise SystemExit(f"cannot open {videos[v]}")
    w = int(caps[sel_views[0]].get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(caps[sel_views[0]].get(cv2.CAP_PROP_FRAME_HEIGHT))
    th = args.target_h
    nw = int(w * th / h)
    rows = 1 if args.no_reproj_row else 2
    out_w, out_h = nw * len(sel_views), th * rows + 34
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (out_w, out_h))

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **k):
            return x

    read_pos = {v: -10 for v in sel_views}
    for idx in tqdm(montage_frames, desc="Montage", unit="frame", ncols=80):
        rp = reproj.get(idx, {v: [] for v in sel_views})
        top_imgs, bot_imgs = [], []
        for v in sel_views:
            cap = caps[v]
            if idx != read_pos[v] + 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            read_pos[v] = idx
            if not ret or frame is None:
                frame = np.zeros((h, w, 3), np.uint8)
            vrp = rp.get(v, [])
            frame = draw_real(frame, preds[v].get(idx, []), vrp,
                              id_to_cidx, args.assign_max_dist)
            # fish-following zoom: crop a square centered on the fish, then upscale.
            box = None
            if args.zoom_crop > 0:
                z = args.zoom_crop
                if vrp:
                    cx = int(np.mean([p[1] for p in vrp])); cy = int(np.mean([p[2] for p in vrp]))
                else:
                    cx, cy = w // 2, h // 2
                x0 = min(max(cx - z // 2, 0), max(0, w - z))
                y0 = min(max(cy - z // 2, 0), max(0, h - z))
                box = (x0, y0, min(z, w), min(z, h))
                frame = frame[y0:y0 + box[3], x0:x0 + box[2]]
            cv2.putText(frame, v, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            top_imgs.append(cv2.resize(frame, (nw, th)))
            if not args.no_reproj_row:
                canvas = np.zeros((h, w, 3), np.uint8)
                canvas = draw_reproj(canvas, vrp, id_to_cidx)
                if box is not None:
                    canvas = canvas[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
                cv2.putText(canvas, f"{v} (3D reproj)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
                bot_imgs.append(cv2.resize(canvas, (nw, th)))
        top_row = np.hstack(top_imgs)
        body = top_row if args.no_reproj_row else np.vstack([top_row, np.hstack(bot_imgs)])
        bar = np.zeros((34, out_w, 3), np.uint8)
        ev_i, mind, center = frame_meta[idx]
        cv2.putText(bar, f"crossing {ev_i}/{len(events)}  t={idx/args.fps:6.1f}s  "
                    f"min-dist={mind:4.1f}mm  frame {idx}",
                    (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        x0 = max(10, out_w - 95 * len(ids) - 30)  # width-aware; fits narrow montages
        for fid in ids:
            cv2.circle(bar, (x0, 17), 7, ID_COLORS[id_to_cidx[fid] % len(ID_COLORS)], -1)
            cv2.putText(bar, f"ID {fid}", (x0 + 12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            x0 += 95
        writer.write(np.vstack([body, bar]))

    writer.release()
    for c in caps.values():
        c.release()
    print(f"Wrote: {args.output}  ({len(events)} crossings, {len(montage_frames)} frames, "
          f"{len(montage_frames)/args.fps:.0f}s)")


if __name__ == "__main__":
    main()
