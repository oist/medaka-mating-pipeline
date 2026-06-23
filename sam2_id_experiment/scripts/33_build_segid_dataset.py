from __future__ import annotations

"""Stage A of the self-supervised appearance-ID pilot: link Detectron2 masks to
concrete SLEAP tracklets and extract per-tracklet crops.

For each frame we match every Detectron2 mask to the concrete-tracklet detections
(from ``<view>.tracklet_bound.csv``) by point-in-mask:

  * exactly ONE tracklet point inside the mask  -> clean single-fish crop, tagged
    with that tracklet_id (the must-link group),
  * ZERO points inside (and none within --gate-px of the mask centroid) -> drop:
    a reflection / false-positive mask,
  * TWO+ points inside -> drop: a MERGED mask (two fish in one blob) -- exactly the
    "anomalous segment" the user asked to exclude.

Per tracklet we also compute the mask-area sequence and drop frames whose area
jumps by more than --max-area-ratio vs the tracklet's running median (a sudden
area spike = partial mask / occlusion). Surviving crops are written with
qc_status='clean_train'.

Output (per view, under --out-dir):
  <view>_crops.h5          images (N, S, S, 3) uint8
  <view>_crop_metadata.parquet   crop_id, frame, tracklet_id, area, area_ratio_prev, qc_status, x, y
  <view>_segid_summary.json      yields + drop reasons

Run (main .venv)::

    python -m sam2_id_experiment.scripts.33_build_segid_dataset \
        --view side1_cam --masks <detectron2_masks.h5> \
        --tracklet-bound-csv <side1_cam.tracklet_bound.csv> \
        --video <cam1_side.mp4> --window 176300 181300 \
        --out-dir runs/segid/<exp>
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from sam2_id_experiment.src.sam2_adapter.mask_io import iter_mask_instances_h5


def link_frame(masks, points, *, gate_px):
    """masks: [(mask_bool, bbox, score)]; points: [(tracklet_id, x, y)].
    Returns [(tracklet_id, mask_idx, status)] for kept masks + drop reasons."""
    assigned, reasons = [], []
    for mi, (m, _bbox, _score) in enumerate(masks):
        H, W = m.shape
        inside = []
        for tid, x, y in points:
            xi, yi = int(round(x)), int(round(y))
            if 0 <= yi < H and 0 <= xi < W and m[yi, xi]:
                inside.append(tid)
        if len(inside) == 1:
            assigned.append((inside[0], mi, "point_inside"))
        elif len(inside) == 0:
            # nearest tracklet point within gate of mask centroid
            ys, xs = np.nonzero(m)
            if len(xs):
                c = np.array([xs.mean(), ys.mean()])
                best, bestd = None, gate_px
                for tid, x, y in points:
                    d = float(np.hypot(x - c[0], y - c[1]))
                    if d < bestd:
                        best, bestd = tid, d
                if best is not None:
                    assigned.append((best, mi, "near_centroid"))
                else:
                    reasons.append("fp_no_tracklet")
            else:
                reasons.append("empty_mask")
        else:
            reasons.append("merged_multi_tracklet")
    return assigned, reasons


def _square_crop(img, bbox, size, pad_frac=0.15, mask=None, background_value=0):
    x0, y0, x1, y1 = [int(v) for v in bbox]
    w, h = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = int(max(w, h) * (1 + pad_frac) / 2) + 1
    H, W = img.shape[:2]
    xa, xb = max(0, int(cx - half)), min(W, int(cx + half))
    ya, yb = max(0, int(cy - half)), min(H, int(cy + half))
    crop = img[ya:yb, xa:xb]
    if crop.size == 0:
        return np.zeros((size, size, 3), np.uint8)
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    if mask is None:
        return crop
    mask_crop = np.asarray(mask[ya:yb, xa:xb], dtype=np.uint8)
    if mask_crop.size == 0:
        return np.full((size, size, 3), int(background_value), np.uint8)
    mask_rs = cv2.resize(mask_crop, (size, size), interpolation=cv2.INTER_NEAREST).astype(bool)
    out = np.full_like(crop, int(background_value))
    out[mask_rs] = crop[mask_rs]
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Link masks to concrete tracklets + extract crops.")
    p.add_argument("--view", required=True)
    p.add_argument("--masks", required=True)
    p.add_argument("--tracklet-bound-csv", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--window", nargs=2, type=int, required=True, metavar=("START", "END"))
    p.add_argument("--out-dir", required=True)
    p.add_argument("--crop-size", type=int, default=128)
    p.add_argument("--gate-px", type=float, default=40.0)
    p.add_argument("--max-area-ratio", type=float, default=2.0)
    p.add_argument("--min-tracklet-crops", type=int, default=5)
    p.add_argument("--mask-background", action="store_true",
                   help="black out pixels outside the matched Detectron2 mask before resizing crops")
    p.add_argument("--mask-background-value", type=int, default=0,
                   help="uint8 value used outside masks when --mask-background is enabled")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    start, end = args.window
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tb = pd.read_csv(args.tracklet_bound_csv)
    tb = tb[(tb.frame >= start) & (tb.frame <= end)]
    points_by_frame: dict[int, list] = defaultdict(list)
    for f, x, y, tid in zip(tb.frame, tb.x, tb.y, tb.tracklet_id):
        points_by_frame[int(f)].append((int(tid), float(x), float(y)))
    print(f"[{args.view}] tracklets in window: {tb.tracklet_id.nunique()}, "
          f"frames with tracklets: {len(points_by_frame)}", flush=True)

    # group masks by frame
    masks_by_frame: dict[int, list] = defaultdict(list)
    for inst in iter_mask_instances_h5(args.masks):
        f = int(inst.frame)
        if start <= f <= end:
            masks_by_frame[f].append((np.asarray(inst.mask).astype(bool),
                                      inst.bbox, float(inst.score)))

    # link + collect crop specs
    crop_specs = []  # (frame, tracklet_id, mask_idx, bbox, area, x, y)
    drop_counter = Counter()
    for f, masks in masks_by_frame.items():
        pts = points_by_frame.get(f, [])
        assigned, reasons = link_frame(masks, pts, gate_px=args.gate_px)
        drop_counter.update(reasons)
        for tid, mi, _status in assigned:
            m, bbox, _score = masks[mi]
            ys, xs = np.nonzero(m)
            area = int(len(xs))
            cx, cy = (float(xs.mean()), float(ys.mean())) if len(xs) else (np.nan, np.nan)
            crop_specs.append((f, tid, mi, bbox, area, cx, cy))

    spec_df = pd.DataFrame(
        crop_specs,
        columns=["frame", "tracklet_id", "mask_idx", "bbox", "area", "x", "y"],
    )
    if spec_df.empty:
        raise SystemExit(f"[{args.view}] no masks linked to tracklets in window")

    # area-spike QC per tracklet (running-median ratio)
    spec_df = spec_df.sort_values(["tracklet_id", "frame"]).reset_index(drop=True)
    spec_df["area_ratio_prev"] = np.nan
    keep_flags = []
    for tid, g in spec_df.groupby("tracklet_id"):
        med = g["area"].expanding().median()
        ratio = np.maximum(g["area"], med) / np.maximum(np.minimum(g["area"], med), 1)
        spec_df.loc[g.index, "area_ratio_prev"] = ratio.values
    spec_df["qc_status"] = np.where(spec_df["area_ratio_prev"] <= args.max_area_ratio,
                                    "clean_train", "reject_area_spike")
    # drop tracklets with too few clean crops
    clean = spec_df[spec_df.qc_status == "clean_train"]
    good_tracklets = clean.groupby("tracklet_id").size()
    good_tracklets = set(good_tracklets[good_tracklets >= args.min_tracklet_crops].index)
    spec_df.loc[~spec_df.tracklet_id.isin(good_tracklets) & (spec_df.qc_status == "clean_train"),
                "qc_status"] = "reject_short_tracklet"

    # extract crops (sequential video read)
    spec_df = spec_df.sort_values("frame").reset_index(drop=True)
    spec_df["crop_id"] = np.arange(len(spec_df))
    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames_needed = {int(f): g.index.tolist() for f, g in spec_df.groupby("frame")}
    images = np.zeros((len(spec_df), args.crop_size, args.crop_size, 3), np.uint8)
    cur = start
    max_f = int(spec_df.frame.max())
    while cur <= max_f:
        ok, img = cap.read()
        if not ok:
            break
        if cur in frames_needed:
            for ridx in frames_needed[cur]:
                bbox = spec_df.loc[ridx, "bbox"]
                mask = None
                if args.mask_background:
                    mask_idx = int(spec_df.loc[ridx, "mask_idx"])
                    mask = masks_by_frame[cur][mask_idx][0]
                images[ridx] = _square_crop(
                    img,
                    bbox,
                    args.crop_size,
                    mask=mask,
                    background_value=args.mask_background_value,
                )
        cur += 1
    cap.release()

    # write outputs
    with h5py.File(out_dir / f"{args.view}_crops.h5", "w") as h:
        h.create_dataset("images", data=images, compression="gzip")
        h.create_dataset("crop_id", data=spec_df["crop_id"].to_numpy())
    meta = spec_df.drop(columns=["bbox", "mask_idx"]).copy()
    meta.to_parquet(out_dir / f"{args.view}_crop_metadata.parquet", index=False)

    n_clean = int((meta.qc_status == "clean_train").sum())
    summary = {
        "view": args.view, "window": [start, end],
        "n_masks_in_window": int(sum(len(v) for v in masks_by_frame.values())),
        "n_linked": int(len(spec_df)),
        "n_clean_train": n_clean,
        "n_clean_tracklets": int(meta[meta.qc_status == "clean_train"].tracklet_id.nunique()),
        "mask_background": bool(args.mask_background),
        "mask_background_value": int(args.mask_background_value),
        "drop_reasons": dict(drop_counter),
        "qc_status_counts": {k: int(v) for k, v in meta.qc_status.value_counts().items()},
        "clean_crops_per_tracklet_median": int(
            meta[meta.qc_status == "clean_train"].groupby("tracklet_id").size().median())
        if n_clean else 0,
    }
    (out_dir / f"{args.view}_segid_summary.json").write_text(json.dumps(summary, indent=2),
                                                            encoding="utf-8")
    print(f"[{args.view}] linked {len(spec_df)} crops; clean_train {n_clean} over "
          f"{summary['n_clean_tracklets']} tracklets; drops {dict(drop_counter)}")
    print(f"[{args.view}] -> {out_dir}")


if __name__ == "__main__":
    main()
