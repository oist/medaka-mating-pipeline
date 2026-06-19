from __future__ import annotations

"""Velocity-standardized inter-fish distance (stIFD) vs the constant 1.5xBL gate.

A constant spatial break threshold (IFD < 1.5*BL) ignores that NN's real failure is
ASSIGNMENT ambiguity, which depends on how far fish move per frame, not raw proximity.
Standardize the gap by the one-step prediction uncertainty:

    sigma_i  = sigma0 + |per-frame displacement of fish i|     (matches the builder's
               last-position NN: a fish's next pos spreads ~its displacement from the last)
    stIFD    = IFD / sqrt(sigma_i^2 + sigma_j^2)                (the closest pair i,j)
    break the tracklet when  stIFD < tau                       (tau in sigma-units)

Medaka move ~2 px/frame, so the constant 1.5*BL=169px gate is wildly over-strict --
stIFD reveals ~90% of frames are NN-confident, not 30%. But the per-frame speed
estimate is noisy, so the raw adaptive threshold flickers and over-fragments; a
rolling-median smoothing of sigma removes the transient. This compares, at MATCHED
coverage, the constant gate vs raw-stIFD vs smoothed-stIFD for each IFD metric. Read-only.

Run::

    python -m scripts.tracking.compare_stifd --slp <top_cam.slp>
"""

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.common.local_paths import choose_path, load_local_paths
from scripts.tracking.concrete_tracklets import build_concrete_tracklets
from scripts.tracking.compare_anchor_tracklets import load_pose_frames, seg_seg_dist

ANCHORS = ["centroid", "mouth", "body-axis"]


def per_det_speed(frames, gate_px):
    """Per-detection centroid speed (px/frame) via NN to the previous frame.
    Unmatched (gap/occlusion) capped at gate_px (treated as fast = uncertain)."""
    speeds = {}
    prev = None
    for f in sorted(frames):
        cur = [d["centroid"] for d in frames[f]]
        sp = []
        for c in cur:
            if prev:
                dd = min(float(np.linalg.norm(c - pc)) for pc in prev)
                sp.append(min(dd, gate_px))
            else:
                sp.append(0.0)
        speeds[f] = sp
        prev = cur
    return speeds


def min_pair(dets, anchor):
    """(distance, i, j) of the closest fish pair under the IFD metric; (inf,-1,-1) if none."""
    best, bi, bj = np.inf, -1, -1
    for i in range(len(dets)):
        for j in range(i + 1, len(dets)):
            if anchor == "mouth":
                if dets[i]["mouth"] is None or dets[j]["mouth"] is None:
                    continue
                d = float(np.linalg.norm(dets[i]["mouth"] - dets[j]["mouth"]))
            elif anchor == "body-axis":
                d = seg_seg_dist(dets[i]["seg"][0], dets[i]["seg"][1],
                                 dets[j]["seg"][0], dets[j]["seg"][1])
            else:
                d = float(np.linalg.norm(dets[i]["centroid"] - dets[j]["centroid"]))
            if d < best:
                best, bi, bj = d, i, j
    return best, bi, bj


def ifd_and_stifd(frames, speeds, anchor, sigma0):
    """Per-frame absolute IFD, raw stIFD, and the pair sigma (for smoothing)."""
    ifd, stifd, sigma = {}, {}, {}
    for f, dets in frames.items():
        d, i, j = min_pair(dets, anchor)
        ifd[f] = d
        if i < 0 or not np.isfinite(d):
            stifd[f], sigma[f] = np.inf, np.nan
        else:
            s = float(np.hypot(sigma0 + speeds[f][i], sigma0 + speeds[f][j]))
            sigma[f], stifd[f] = s, d / s
    return ifd, stifd, sigma


def smooth_by_frame(series, w):
    """Rolling-median smoothing over the frame index (finite entries only)."""
    fs = sorted(series)
    vals = np.array([series[f] for f in fs], dtype=float)
    half = w // 2
    out = {}
    for i, f in enumerate(fs):
        if not np.isfinite(vals[i]):
            out[f] = vals[i]
            continue
        win = vals[max(0, i - half):i + half + 1]
        win = win[np.isfinite(win)]
        out[f] = float(np.median(win)) if win.size else vals[i]
    return out


def sweep_gate(fd, metric, thresholds, common):
    out = []
    for th in thresholds:
        _, st = build_concrete_tracklets(fd, min_sep_by_frame=metric, sep_threshold=th, **common)
        out.append((th, 100 * st["coverage_frac_of_fish_frames"],
                    st["n_fragments"], st["frag_len_frames_median"]))
    return out


def at_target(curve, target):
    return min(curve, key=lambda r: abs(r[1] - target))


def _fd(frames, anchor):
    if anchor == "mouth":
        return {f: [(d["mouth"][0], d["mouth"][1], d["score"]) for d in ds if d["mouth"] is not None]
                for f, ds in frames.items()}
    return {f: [(d["centroid"][0], d["centroid"][1], d["score"]) for d in ds]
            for f, ds in frames.items()}


def build_parser():
    p = argparse.ArgumentParser(description="stIFD (raw + smoothed) vs constant 1.5xBL gate.")
    p.add_argument("experiment", nargs="?", default=None)
    p.add_argument("--view", default="top_cam")
    p.add_argument("--slp", default=None)
    p.add_argument("--num-fish", type=int, default=3)
    p.add_argument("--sigma0", type=float, default=2.0, help="localization-noise floor (px)")
    p.add_argument("--smooth-window", type=int, default=9, help="frames for rolling-median sigma smoothing")
    p.add_argument("--targets", default="50,90", help="coverage %% levels for the matched comparison")
    p.add_argument("--jump-bl", type=float, default=1.0)
    p.add_argument("--min-score", type=float, default=0.3)
    p.add_argument("--max-gap", type=int, default=5)
    p.add_argument("--paths-config", default=None)
    p.add_argument("--sleap-output-dir", default=None)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    slp = Path(args.slp) if args.slp else None
    if slp is None:
        if args.experiment is None:
            raise SystemExit("pass an experiment name or --slp")
        local = load_local_paths(args.paths_config)
        out = choose_path(args.sleap_output_dir, local.sleap_output_dir if local else None)
        slp = Path(out) / args.experiment / f"{args.view}.slp"
    if not slp.exists():
        raise SystemExit(f"slp not found: {slp}")
    targets = [float(t) for t in args.targets.split(",")]

    print(f"loading {slp} ...", flush=True)
    frames, bl, _ = load_pose_frames(slp, num_fish=args.num_fish, top_k=True)
    speeds = per_det_speed(frames, gate_px=bl)
    mean_sp = float(np.mean([s for sp in speeds.values() for s in sp]))
    print(f"frames={len(frames)}  body_length~{bl:.1f}px  mean speed~{mean_sp:.1f}px/frame  "
          f"sigma0={args.sigma0}px  smooth_window={args.smooth_window}")

    common = dict(num_fish=args.num_fish, body_length=bl, jump_bl=args.jump_bl,
                  min_score=args.min_score, max_gap=args.max_gap)
    const_bls = [0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5]
    taus = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0, 12.0]

    for anchor in ANCHORS:
        fd = _fd(frames, anchor)
        ifd, stifd_raw, sigma = ifd_and_stifd(frames, speeds, anchor, args.sigma0)
        sigma_s = smooth_by_frame(sigma, args.smooth_window)
        stifd_sm = {f: (ifd[f] / sigma_s[f] if np.isfinite(sigma_s.get(f, np.nan)) and sigma_s[f] > 0
                        else np.inf) for f in ifd}

        c_const = sweep_gate(fd, ifd, [b * bl for b in const_bls], common)
        c_raw = sweep_gate(fd, stifd_raw, taus, common)
        c_sm = sweep_gate(fd, stifd_sm, taus, common)
        print(f"\n--- anchor: {anchor}  (frags / median fr at matched coverage) ---")
        print(f"  {'target':>7}{'constant':>20}{'stIFD raw':>18}{'stIFD smooth':>18}")
        for tgt in targets:
            k = at_target(c_const, tgt)
            r = at_target(c_raw, tgt)
            s = at_target(c_sm, tgt)
            print(f"  {tgt:>6.0f}%"
                  f"  {k[2]:>6}/{k[3]:<3}@{k[1]:>4.0f}%"
                  f"  {r[2]:>6}/{r[3]:<3}@{r[1]:>4.0f}%"
                  f"  {s[2]:>6}/{s[3]:<3}@{s[1]:>4.0f}%")


if __name__ == "__main__":
    main()
