from __future__ import annotations

"""Compare concrete-tracklet anchors: centroid vs mouth vs body-axis (vector).

Same conservative tracklet builder, three notions of where a fish IS and when two
fish are RESOLVABLE:

  * centroid  -- mean of visible nodes; swap-invariant but low-res (two fish nose-to-
                 tail can have near-equal centroids while their bodies are clearly apart).
  * mouth     -- the mouth keypoint; high-res, but NaN when the mouth is missed and it
                 JUMPS to the tail whenever SLEAP swaps the mouth/tail labels.
  * body-axis -- separation = distance between the two fish's mouth->tail SEGMENTS
                 (the "vector"). Swap-invariant (a segment is the same either way) AND
                 high-res: two fish are "separated" iff their bodies don't overlap, even
                 when centroids are close. Linking still uses the centroid.

For each, reports clean-frame %, fragment count, median fragment length and coverage.
Also reports the mouth/tail label-SWAP rate (heading flips within a swap-invariant
centroid tracklet) -- the robustness cost mouth-anchoring pays. Read-only.

Run::

    python -m scripts.tracking.compare_anchor_tracklets --slp <top_cam.slp>
"""

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.common.local_paths import choose_path, load_local_paths
from scripts.tracking.concrete_tracklets import build_concrete_tracklets


def _pt_seg(p, a, b):
    ab = b - a
    denom = float(ab.dot(ab))
    t = float(np.clip((p - a).dot(ab) / denom, 0.0, 1.0)) if denom > 1e-12 else 0.0
    return float(np.linalg.norm(p - (a + t * ab)))


def _ccw(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])


def _seg_intersect(p1, p2, q1, q2):
    d1, d2 = _ccw(q1, q2, p1), _ccw(q1, q2, p2)
    d3, d4 = _ccw(p1, p2, q1), _ccw(p1, p2, q2)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def seg_seg_dist(p1, p2, q1, q2):
    """Min distance between 2D segments [p1,p2] and [q1,q2] (0 if they cross)."""
    p1, p2, q1, q2 = (np.asarray(v, float) for v in (p1, p2, q1, q2))
    if _seg_intersect(p1, p2, q1, q2):
        return 0.0
    return min(_pt_seg(p1, q1, q2), _pt_seg(p2, q1, q2),
               _pt_seg(q1, p1, p2), _pt_seg(q2, p1, p2))


def _min_pair_seg(dets):
    n = len(dets)
    if n < 2:
        return np.inf
    m = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            m = min(m, seg_seg_dist(dets[i]["seg"][0], dets[i]["seg"][1],
                                    dets[j]["seg"][0], dets[j]["seg"][1]))
    return m


def load_pose_frames(slp_path, *, num_fish, top_k):
    import sleap_io as sio
    labels = sio.load_slp(str(slp_path))
    nodes = [n.name for n in labels.skeletons[0].nodes]
    mi = nodes.index("mouth") if "mouth" in nodes else 0
    ti = nodes.index("tail") if "tail" in nodes else (1 if len(nodes) > 1 else 0)
    frames, bls = {}, []
    n_mouth_nan = n_det = 0
    for f in sorted(labels.labeled_frames, key=lambda x: x.frame_idx):
        dets = []
        for inst in f.instances:
            pn = inst.numpy()[:, :2]
            fin = np.isfinite(pn).all(axis=1)
            if not fin.any():
                continue
            centroid = pn[fin].mean(axis=0)
            mouth = pn[mi] if mi < len(pn) and np.isfinite(pn[mi]).all() else None
            tail = pn[ti] if ti < len(pn) and np.isfinite(pn[ti]).all() else None
            score = getattr(inst, "score", None)
            score = float(score) if score is not None and np.isfinite(score) else 0.0
            seg_a = mouth if mouth is not None else centroid
            seg_b = tail if tail is not None else centroid
            dets.append({"centroid": centroid, "mouth": mouth, "tail": tail,
                         "score": score, "seg": (np.asarray(seg_a, float), np.asarray(seg_b, float))})
            if mouth is not None and tail is not None:
                bls.append(float(np.linalg.norm(mouth - tail)))
        if top_k and len(dets) > num_fish:
            dets = sorted(dets, key=lambda d: d["score"], reverse=True)[:num_fish]
        for d in dets:
            n_det += 1
            if d["mouth"] is None:
                n_mouth_nan += 1
        frames[int(f.frame_idx)] = dets
    bl = float(np.median(bls)) if bls else float("nan")
    return frames, bl, (n_mouth_nan / max(n_det, 1))


def heading_flip_rate(frames, tracklets):
    """Within swap-invariant centroid tracklets, how often does the mouth->tail
    heading flip >120 deg between consecutive frames (= a likely mouth/tail label swap)."""
    head = {}
    for f, dets in frames.items():
        hs = []
        for d in dets:
            if d["mouth"] is not None and d["tail"] is not None:
                v = d["tail"] - d["mouth"]
                n = np.linalg.norm(v)
                hs.append((d["centroid"], v / n if n > 1e-6 else None))
        head[f] = hs

    def nearest_heading(f, xy):
        best, bd = None, np.inf
        for c, h in head.get(f, []):
            dd = np.hypot(c[0] - xy[0], c[1] - xy[1])
            if dd < bd:
                bd, best = dd, h
        return best

    flips = trans = 0
    for t in tracklets:
        pts = t["points"]
        for (f0, x0, y0, _), (f1, x1, y1, _) in zip(pts, pts[1:]):
            if f1 != f0 + 1:
                continue
            h0, h1 = nearest_heading(f0, (x0, y0)), nearest_heading(f1, (x1, y1))
            if h0 is None or h1 is None:
                continue
            trans += 1
            if float(np.dot(h0, h1)) < -0.5:   # >120 deg
                flips += 1
    return flips / max(trans, 1), trans


def build_parser():
    p = argparse.ArgumentParser(description="Compare centroid / mouth / body-axis tracklet anchors.")
    p.add_argument("experiment", nargs="?", default=None)
    p.add_argument("--view", default="top_cam")
    p.add_argument("--slp", default=None)
    p.add_argument("--num-fish", type=int, default=3)
    p.add_argument("--sep-bl", type=float, default=1.5)
    p.add_argument("--jump-bl", type=float, default=1.0)
    p.add_argument("--min-score", type=float, default=0.3)
    p.add_argument("--max-gap", type=int, default=5)
    p.add_argument("--fps", type=float, default=166.0)
    p.add_argument("--sweep", action="store_true",
                   help="also sweep sep-bl per mode (fair comparison across thresholds)")
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

    print(f"loading {slp} ...", flush=True)
    frames, bl, mouth_nan = load_pose_frames(slp, num_fish=args.num_fish, top_k=True)
    print(f"frames={len(frames)}  body_length~{bl:.1f}px  mouth-NaN rate={100*mouth_nan:.1f}%")

    common = dict(num_fish=args.num_fish, body_length=bl, sep_bl=args.sep_bl,
                  jump_bl=args.jump_bl, min_score=args.min_score, max_gap=args.max_gap)
    fd_centroid = {f: [(d["centroid"][0], d["centroid"][1], d["score"]) for d in ds]
                   for f, ds in frames.items()}
    fd_mouth = {f: [(d["mouth"][0], d["mouth"][1], d["score"]) for d in ds if d["mouth"] is not None]
                for f, ds in frames.items()}
    seg_sep = {f: _min_pair_seg(ds) for f, ds in frames.items()}

    tr_c, st_c = build_concrete_tracklets(fd_centroid, **common)
    tr_m, st_m = build_concrete_tracklets(fd_mouth, **common)
    tr_v, st_v = build_concrete_tracklets(fd_centroid, min_sep_by_frame=seg_sep, **common)
    flip_rate, n_trans = heading_flip_rate(frames, tr_c)

    print(f"\nmouth/tail label-SWAP proxy: {100*flip_rate:.1f}% of consecutive frames flip heading "
          f">120 deg within a centroid tracklet ({n_trans} transitions checked)")
    print(f"\n{'mode':<11}{'clean%':>8}{'fragments':>11}{'med_frag_fr':>13}{'coverage%':>11}")
    for name, st in [("centroid", st_c), ("mouth", st_m), ("body-axis", st_v)]:
        print(f"{name:<11}{100*st['clean_frac']:>8.1f}{st['n_fragments']:>11}"
              f"{st['frag_len_frames_median']:>13}{100*st['coverage_frac_of_fish_frames']:>11.1f}")
    print("\nnotes: centroid = swap-invariant, low-res | mouth = high-res but NaN+swap-prone | "
          "body-axis = swap-invariant + body-overlap separation")

    if args.sweep:
        print("\n=== sep-bl sweep  (clean% / fragments / med_frag_fr) ===")
        print(f"{'sep_bl':>7}{'centroid':>20}{'mouth':>20}{'body-axis':>20}")
        for s in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
            cc = dict(num_fish=args.num_fish, body_length=bl, sep_bl=s,
                      jump_bl=args.jump_bl, min_score=args.min_score, max_gap=args.max_gap)
            _, a = build_concrete_tracklets(fd_centroid, **cc)
            _, b = build_concrete_tracklets(fd_mouth, **cc)
            _, c = build_concrete_tracklets(fd_centroid, min_sep_by_frame=seg_sep, **cc)

            def fmt(st):
                return f"{100*st['clean_frac']:.0f}%/{st['n_fragments']}/{st['frag_len_frames_median']}"
            print(f"{s:>7}{fmt(a):>20}{fmt(b):>20}{fmt(c):>20}")


if __name__ == "__main__":
    main()
