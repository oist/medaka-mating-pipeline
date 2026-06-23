from __future__ import annotations

"""Overlay Detectron2 instance masks and SLEAP predictions on the same frames, for
top and side views, and quantify how much they agree.

Detectron2 masks are read from an existing ``detectron2_masks.h5`` (no Detectron2 /
GPU needed -- inference already ran). SLEAP predictions come from the per-view .slp.
For a sample of frames it draws masks (filled, per-instance colour) + SLEAP skeleton
(yellow) on the video frame, writes a montage PNG per view, and computes an
agreement panel:

  * sleap_in_mask_rate  -- fraction of SLEAP instances whose body falls inside a mask
                           (SLEAP fish that segmentation also found),
  * mask_has_sleap_rate -- fraction of masks that contain a SLEAP keypoint
                           (1 - this = false-positive / reflection masks),
  * count_agreement     -- fraction of frames with #masks == #SLEAP instances,
  * centroid_dist_px    -- median SLEAP-centroid to mask-centroid distance for matches.

Run (main .venv; needs opencv, h5py, sleap-io, matplotlib)::

    python -m sam2_id_experiment.scripts.32_overlay_detectron2_vs_sleap \
        --view top_cam --video <cam0_top.mp4> \
        --masks <detectron2_masks.h5> --slp <top_cam.slp> \
        --out-dir runs/seg_qc/<exp> --n-sample 8
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_COLORS = [(31, 119, 180), (214, 39, 40), (44, 160, 44), (148, 103, 189),
           (255, 127, 14), (140, 86, 75)]


# ------------------------------- mask loading ------------------------------- #

def load_masks_for_frames(masks_h5: str, frames: list[int]) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    with h5py.File(masks_h5, "r") as h:
        for f in frames:
            key = f"frame_{f:06d}"
            if key not in h:
                out[f] = []
                continue
            insts = []
            g = h[key]
            for ik in sorted(k for k in g.keys() if k.startswith("instance_")):
                grp = g[ik]
                m = grp["mask"][()].astype(bool)
                insts.append({"mask": m, "score": float(grp.attrs.get("score", np.nan))})
            out[f] = insts
    return out


def mask_frame_range(masks_h5: str) -> list[int]:
    with h5py.File(masks_h5, "r") as h:
        return sorted(int(k.split("_")[1]) for k in h.keys() if k.startswith("frame_"))


# ------------------------------- sleap loading ------------------------------ #

def load_sleap_instances(slp: str, frames: set[int]):
    """Return {frame: [ (n_nodes,2) xy array, ... ]} for the requested frames, plus
    skeleton edges. Uses sleap-io; only keeps the requested frames."""
    import sleap_io as sio
    labels = sio.load_slp(slp)
    edges = []
    if labels.skeletons:
        sk = labels.skeletons[0]
        try:
            edges = list(sk.edge_inds)
        except Exception:
            edges = []
    per_frame: dict[int, list[np.ndarray]] = {}
    for lf in labels.labeled_frames:
        fi = int(lf.frame_idx)
        if fi not in frames:
            continue
        insts = []
        for inst in lf.instances:
            pts = inst.numpy()  # (n_nodes, 2), nan for missing
            insts.append(np.asarray(pts, dtype=float))
        per_frame[fi] = insts
    return per_frame, edges


def _centroid(xy: np.ndarray) -> np.ndarray:
    fin = np.isfinite(xy).all(axis=1)
    return np.nanmean(xy[fin], axis=0) if fin.any() else np.array([np.nan, np.nan])


# ------------------------------- metrics ------------------------------------ #

def frame_agreement(masks: list[dict], sleap: list[np.ndarray]) -> dict:
    n_masks, n_sleap = len(masks), len(sleap)
    mask_arrs = [m["mask"] for m in masks]
    H, W = (mask_arrs[0].shape if mask_arrs else (0, 0))

    def inside_any(x, y):
        if not mask_arrs or not (0 <= int(y) < H and 0 <= int(x) < W):
            return -1
        for mi, m in enumerate(mask_arrs):
            if m[int(y), int(x)]:
                return mi
        return -1

    # SLEAP instance matched if its centroid (or any node) lands in a mask
    sleap_matched = 0
    centroid_dists = []
    mask_hit = [False] * len(mask_arrs)
    for xy in sleap:
        c = _centroid(xy)
        hit = -1
        if np.isfinite(c).all():
            hit = inside_any(c[0], c[1])
        if hit < 0:  # fall back to any visible node
            for p in xy:
                if np.isfinite(p).all():
                    hit = inside_any(p[0], p[1])
                    if hit >= 0:
                        break
        if hit >= 0:
            sleap_matched += 1
            mask_hit[hit] = True
            mc = _mask_centroid(mask_arrs[hit])
            if np.isfinite(c).all():
                centroid_dists.append(float(np.linalg.norm(c - mc)))
    return {
        "n_masks": n_masks, "n_sleap": n_sleap,
        "sleap_matched": sleap_matched,
        "mask_with_sleap": int(sum(mask_hit)),
        "centroid_dists": centroid_dists,
        "count_agree": int(n_masks == n_sleap and n_sleap > 0),
    }


def _mask_centroid(m: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(m)
    return np.array([xs.mean(), ys.mean()]) if len(xs) else np.array([np.nan, np.nan])


# ------------------------------- rendering ---------------------------------- #

def overlay_frame(img, masks, sleap, edges, alpha=0.45):
    out = img.copy()
    for mi, m in enumerate(masks):
        color = _COLORS[mi % len(_COLORS)]
        msk = m["mask"]
        colored = np.zeros_like(out)
        colored[msk] = color
        out = np.where(msk[..., None], (alpha * colored + (1 - alpha) * out).astype(np.uint8), out)
        cnts, _ = cv2.findContours(msk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, color, 2)
    for xy in sleap:
        for (a, b) in edges:
            if a < len(xy) and b < len(xy) and np.isfinite(xy[a]).all() and np.isfinite(xy[b]).all():
                cv2.line(out, tuple(xy[a].astype(int)), tuple(xy[b].astype(int)), (0, 255, 255), 2)
        for p in xy:
            if np.isfinite(p).all():
                cv2.circle(out, tuple(p.astype(int)), 4, (0, 255, 255), -1)
                cv2.circle(out, tuple(p.astype(int)), 4, (0, 0, 0), 1)
    return out


def crop_to_content(img, masks, sleap, pad=120):
    pts = []
    for m in masks:
        ys, xs = np.nonzero(m["mask"])
        if len(xs):
            pts.append([xs.min(), ys.min()]); pts.append([xs.max(), ys.max()])
    for xy in sleap:
        fin = xy[np.isfinite(xy).all(axis=1)]
        pts.extend(fin.tolist())
    if not pts:
        return img
    pts = np.array(pts)
    H, W = img.shape[:2]
    x0 = max(0, int(pts[:, 0].min()) - pad); x1 = min(W, int(pts[:, 0].max()) + pad)
    y0 = max(0, int(pts[:, 1].min()) - pad); y1 = min(H, int(pts[:, 1].max()) + pad)
    return img[y0:y1, x0:x1]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Overlay Detectron2 masks vs SLEAP predictions + agreement.")
    p.add_argument("--view", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--masks", required=True, help="detectron2_masks.h5")
    p.add_argument("--slp", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-sample", type=int, default=8)
    p.add_argument("--frames", nargs="*", type=int, default=None, help="explicit frames (else evenly sampled)")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    avail = mask_frame_range(args.masks)
    if not avail:
        raise SystemExit(f"no frames in {args.masks}")
    if args.frames:
        sample = [f for f in args.frames if f in set(avail)]
    else:
        idx = np.linspace(0, len(avail) - 1, args.n_sample).astype(int)
        sample = [avail[i] for i in idx]
    print(f"[{args.view}] mask frames {avail[0]}-{avail[-1]} ({len(avail)}); sampling {sample}", flush=True)

    masks_by_frame = load_masks_for_frames(args.masks, sample)
    print(f"[{args.view}] loading SLEAP {args.slp} ...", flush=True)
    sleap_by_frame, edges = load_sleap_instances(args.slp, set(sample))

    cap = cv2.VideoCapture(args.video)
    panels, per_frame_metrics = [], []
    for f in sample:
        masks = masks_by_frame.get(f, [])
        sleap = sleap_by_frame.get(f, [])
        m = frame_agreement(masks, sleap)
        m["frame"] = f
        per_frame_metrics.append(m)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        ov = overlay_frame(img, masks, sleap, edges)
        ov = crop_to_content(ov, masks, sleap)
        panels.append((f, ov, m))
    cap.release()

    # montage
    if panels:
        ncol = min(4, len(panels))
        nrow = int(np.ceil(len(panels) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.2 * nrow), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for (f, ov, m), ax in zip(panels, axes.flat):
            ax.imshow(cv2.cvtColor(ov, cv2.COLOR_BGR2RGB))
            ax.set_title(f"f{f}  masks={m['n_masks']} sleap={m['n_sleap']}  "
                         f"in-mask {m['sleap_matched']}/{m['n_sleap']}", fontsize=8)
        fig.suptitle(f"{args.view}: Detectron2 masks (filled) vs SLEAP (yellow)", fontsize=11)
        fig.tight_layout()
        png = out_dir / f"{args.view}_detectron2_vs_sleap.png"
        fig.savefig(png, dpi=110)
        plt.close(fig)
        print(f"[{args.view}] wrote {png}")

    # aggregate metrics
    tot_sleap = sum(m["n_sleap"] for m in per_frame_metrics)
    tot_masks = sum(m["n_masks"] for m in per_frame_metrics)
    tot_sleap_matched = sum(m["sleap_matched"] for m in per_frame_metrics)
    tot_mask_with_sleap = sum(m["mask_with_sleap"] for m in per_frame_metrics)
    dists = [d for m in per_frame_metrics for d in m["centroid_dists"]]
    summary = {
        "view": args.view, "n_frames_sampled": len(per_frame_metrics),
        "mask_frame_range": [avail[0], avail[-1]], "n_mask_frames": len(avail),
        "mean_masks_per_frame": round(tot_masks / max(len(per_frame_metrics), 1), 2),
        "mean_sleap_per_frame": round(tot_sleap / max(len(per_frame_metrics), 1), 2),
        "sleap_in_mask_rate": round(tot_sleap_matched / max(tot_sleap, 1), 4),
        "mask_has_sleap_rate": round(tot_mask_with_sleap / max(tot_masks, 1), 4),
        "count_agreement": round(np.mean([m["count_agree"] for m in per_frame_metrics]), 4)
        if per_frame_metrics else float("nan"),
        "median_centroid_dist_px": round(float(np.median(dists)), 2) if dists else None,
        "per_frame": per_frame_metrics,
    }
    (out_dir / f"{args.view}_agreement.json").write_text(json.dumps(summary, indent=2, default=str),
                                                         encoding="utf-8")
    print(f"[{args.view}] sleap_in_mask={summary['sleap_in_mask_rate']}  "
          f"mask_has_sleap={summary['mask_has_sleap_rate']}  "
          f"count_agree={summary['count_agreement']}  median_dist={summary['median_centroid_dist_px']}px")


if __name__ == "__main__":
    main()
