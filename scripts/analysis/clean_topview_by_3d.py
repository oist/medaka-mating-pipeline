"""Remove surface-reflection phantoms from a single-view (top-cam) video by
keeping only the image blobs that sit at 3D-verified fish positions.

Motivation
----------
A top-down camera of the tank shows each fish AND a fish-shaped water-surface
reflection. Single-view trackers (idtracker.ai, UDMT) segment by intensity and
cannot tell a real fish from its reflection, which wrecks global identity
propagation. Our 3D pipeline already rejects reflections via cross-view geometry,
so we can use the reprojected ``final_fish_id`` positions to delete every blob
that does NOT correspond to a real 3D fish. The cleaned video has ~N clean blobs
per frame, which lets idtracker.ai's (validated ~99% accurate) appearance model
accumulate identities across the whole clip.

For each frame in ``[--start-frame, --end-frame)`` (full-video frame indices,
matching the 3D CSV), we:
  1. reproject the finite 3D fish positions for that frame into the chosen camera,
  2. segment blobs via median-background absolute difference,
  3. keep blobs whose centroid is within ``--gate-px`` of any reprojected fish,
  4. paint everything else to the median background.

Output is a video with identical frame indexing (frame 0 == --start-frame).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from scripts.reconstruction_3d.calibration import Calibration

# top=cam0, side1=cam1, side2=cam2 (matches the rest of the pipeline).
DEFAULT_CAM_INDEX = 0


def _median_background(cap, start: int, end: int, n_samples: int) -> np.ndarray:
    idxs = np.linspace(start, end - 1, n_samples).astype(int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    if not frames:
        raise SystemExit("could not read any frames for background")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def _seek_to(cap, target: int) -> None:
    """Position ``cap`` so the next ``read()`` returns frame ``target``.

    Robust to imprecise keyframe seeking on long/variable-GOP HEVC: after the
    absolute seek we grab-forward to the exact frame using the reported position
    (cf. ``scripts/visualization/render_4panel_video.py``). On this machine's
    build cv2 seeking is frame-exact (verified), so the grab loop is usually a
    no-op, but this keeps the frame<->reproj mapping correct on other builds.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    landed = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))
    if landed > target:  # overshoot — restart and grab forward from the start
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        landed = 0
    while landed < target:
        if not cap.grab():
            break
        landed += 1


def _reproj_by_frame(csv_3d: Path, cal: Calibration, cam: int) -> dict[int, np.ndarray]:
    df = pd.read_csv(csv_3d, usecols=lambda c: c in
                     ("frame", "final_fish_id", "fish_id", "X", "Y", "Z"))
    df = df.dropna(subset=["X", "Y", "Z"])
    if df.empty:
        raise SystemExit(f"no finite XYZ rows in {csv_3d}")
    uv = cal.compute_imCoords_from_XYZ(df[["X", "Y", "Z"]].to_numpy(float))
    xy = uv[:, cam, :]
    frames = df["frame"].to_numpy().astype(int)
    out: dict[int, list] = {}
    for k in range(len(df)):
        x, y = xy[k, 0], xy[k, 1]
        if np.isfinite(x) and np.isfinite(y):
            out.setdefault(int(frames[k]), []).append((float(x), float(y)))
    return {f: np.array(v, dtype=float) for f, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--csv-3d", type=Path, required=True)
    ap.add_argument("--model-folder", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--start-frame", type=int, required=True)
    ap.add_argument("--end-frame", type=int, required=True, help="exclusive")
    ap.add_argument("--cam-index", type=int, default=DEFAULT_CAM_INDEX)
    ap.add_argument("--gate-px", type=float, default=70.0,
                    help="keep a blob if its centroid is within this many px of "
                         "any reprojected fish position.")
    ap.add_argument("--diff-ths", type=int, default=35,
                    help="median-background abs-difference threshold for blobs.")
    ap.add_argument("--area-min", type=float, default=300.0)
    ap.add_argument("--area-max", type=float, default=20000.0)
    ap.add_argument("--bg-samples", type=int, default=60)
    ap.add_argument("--fps", type=float, default=166.0)
    args = ap.parse_args()

    cal = Calibration(args.model_folder)
    reproj = _reproj_by_frame(args.csv_3d, cal, args.cam_index)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end = min(args.end_frame, total)
    bg = _median_background(cap, args.start_frame, end, args.bg_samples)
    bg_bgr = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
    h, w = bg.shape

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (w, h))
    kernel = np.ones((3, 3), np.uint8)

    try:
        from tqdm import tqdm
        rng = tqdm(range(args.start_frame, end), unit="frame", ncols=80)
    except ImportError:
        rng = range(args.start_frame, end)

    _seek_to(cap, args.start_frame)
    n_written = n_with3 = 0
    kept_counts = []
    for f in rng:
        ok, frame = cap.read()
        if not ok:
            # Distinguish true EOF from a transient decode hiccup: at/near the
            # end, stop; otherwise write a background (no-fish) frame so output
            # frame index stays aligned to (f - start_frame).
            if f >= end - 1 or int(cap.get(cv2.CAP_PROP_POS_FRAMES)) >= total:
                break
            writer.write(bg_bgr.copy())
            n_written += 1
            kept_counts.append(0)
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, bg)
        mask = (diff > args.diff_ths).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        nlab, lab, stats, cent = cv2.connectedComponentsWithStats(mask)
        pos = reproj.get(f)
        keep = np.zeros((h, w), np.uint8)
        keep_labels: set[int] = set()
        if pos is not None and len(pos):
            # Area-valid components only.
            valid = [(li, float(cent[li][0]), float(cent[li][1]))
                     for li in range(1, nlab)
                     if args.area_min <= stats[li, cv2.CC_STAT_AREA] <= args.area_max]
            # Keep, per real fish, ONLY its single nearest blob within the gate.
            # This caps output at <=len(pos) blobs/frame and structurally excludes
            # reflections (never the nearest blob to a 3D-verified fish position).
            for fx, fy in pos:
                best_li, best_d = None, args.gate_px
                for li, cx, cy in valid:
                    d = float(np.hypot(fx - cx, fy - cy))
                    if d <= best_d:
                        best_d, best_li = d, li
                if best_li is not None:
                    keep_labels.add(best_li)
            for li in keep_labels:
                keep[lab == li] = 1
        n_kept = len(keep_labels)
        kept_counts.append(n_kept)
        if n_kept >= 3:
            n_with3 += 1
        out = bg_bgr.copy()
        m3 = keep.astype(bool)
        out[m3] = frame[m3]
        writer.write(out)
        n_written += 1

    writer.release()
    cap.release()
    kc = np.array(kept_counts) if kept_counts else np.array([0])
    print(f"Wrote {n_written} frames -> {args.output}")
    print(f"kept blobs/frame: mean={kc.mean():.2f} median={np.median(kc):.0f} "
          f"%frames>=3={100*n_with3/max(n_written,1):.1f}%  "
          f"(==3: {100*(kc==3).mean():.1f}%, <=3: {100*(kc<=3).mean():.1f}%)")


if __name__ == "__main__":
    main()
