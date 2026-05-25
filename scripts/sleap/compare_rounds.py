from __future__ import annotations

"""
Compare round-N vs round-(N-1) SLEAP inference on the same experiment.

Reads two parallel `.slp` files per view (round-A vs round-B) and reports
metrics that distinguish "did the model find more / better-anchored fish":

* Frames with detections, total instances detected
* Per-frame instance-count histogram (we expect 3 fish: count==3 is good)
* Mean instance prediction score (centroid model confidence)
* Mean per-point visibility score (centered-instance confidence)
* Mouth-tail pixel distance distribution (body-length proxy; node-swaps
  show up as outliers far below the bulk distribution)
* Per-view summary table

Usage::

    python -m scripts.sleap.compare_rounds \
        --label-a round3_3node_full \
        --dir-a Z:/.../sleap_output/<exp>/round3_3node_full \
        --label-b round4_260520 \
        --dir-b Z:/.../sleap_output/<exp>/round4_260520

Both directories must contain ``{top_cam,side1_cam,side2_cam}.slp``.
"""

import argparse
import math
from collections import Counter
from pathlib import Path

import numpy as np
import sleap_io as sio


VIEWS = ("top_cam", "side1_cam", "side2_cam")


def summarise_slp(slp_path: Path) -> dict:
    """Aggregate stats for a single .slp file."""
    labels = sio.load_slp(str(slp_path))
    n_frames_total = 0
    n_frames_with = 0
    n_instances = 0
    inst_count_hist: Counter[int] = Counter()
    inst_scores: list[float] = []
    pt_scores: list[float] = []
    mouth_tail_dists: list[float] = []
    node_names = [n.name for n in labels.skeletons[0].nodes]
    mouth_idx = node_names.index("mouth") if "mouth" in node_names else None
    tail_idx = node_names.index("tail") if "tail" in node_names else None

    for lf in labels.labeled_frames:
        n_frames_total += 1
        n = len(lf.instances)
        inst_count_hist[n] += 1
        if n > 0:
            n_frames_with += 1
        n_instances += n
        for inst in lf.instances:
            s = getattr(inst, "score", None)
            if s is not None and not math.isnan(s):
                inst_scores.append(float(s))
            # Per-point visibility / score
            pts = inst.numpy()  # (n_nodes, 2 or 3)
            if mouth_idx is not None and tail_idx is not None and pts.shape[0] > max(mouth_idx, tail_idx):
                m_xy = pts[mouth_idx, :2]
                t_xy = pts[tail_idx, :2]
                if not (np.isnan(m_xy).any() or np.isnan(t_xy).any()):
                    d = float(np.linalg.norm(m_xy - t_xy))
                    mouth_tail_dists.append(d)
            # pred-point score: each predicted point has .score
            for p in inst.points:
                sc = p.get("score") if isinstance(p, dict) else getattr(p, "score", None)
                if sc is None:
                    # Newer sleap-io: predicted points are a np.void array with score field
                    try:
                        sc = float(p["score"])
                    except Exception:
                        sc = None
                if sc is not None and not math.isnan(sc):
                    pt_scores.append(float(sc))

    return {
        "path": str(slp_path),
        "node_names": node_names,
        "n_frames_total": n_frames_total,
        "n_frames_with": n_frames_with,
        "n_instances": n_instances,
        "inst_count_hist": dict(inst_count_hist),
        "mean_inst_score": float(np.mean(inst_scores)) if inst_scores else float("nan"),
        "mean_pt_score": float(np.mean(pt_scores)) if pt_scores else float("nan"),
        "mouth_tail": {
            "n": len(mouth_tail_dists),
            "mean": float(np.mean(mouth_tail_dists)) if mouth_tail_dists else float("nan"),
            "p10": float(np.percentile(mouth_tail_dists, 10)) if mouth_tail_dists else float("nan"),
            "p50": float(np.percentile(mouth_tail_dists, 50)) if mouth_tail_dists else float("nan"),
            "p90": float(np.percentile(mouth_tail_dists, 90)) if mouth_tail_dists else float("nan"),
            "frac_lt_10": float(np.mean(np.array(mouth_tail_dists) < 10.0)) if mouth_tail_dists else float("nan"),
        },
    }


def print_summary(label: str, view: str, s: dict) -> None:
    h = s["inst_count_hist"]
    total = sum(h.values()) or 1
    pct = {k: 100.0 * h.get(k, 0) / total for k in (0, 1, 2, 3, 4, 5, 6)}
    print(f"  [{label}] {view}: n_frames={s['n_frames_total']} "
          f"n_inst={s['n_instances']} nodes={s['node_names']}")
    print(f"    inst-per-frame: "
          + "  ".join(f"{k}={pct[k]:.1f}%" for k in (0, 1, 2, 3, 4, 5, 6)))
    print(f"    mean instance score: {s['mean_inst_score']:.3f}  "
          f"mean point score: {s['mean_pt_score']:.3f}")
    mt = s["mouth_tail"]
    print(f"    mouth->tail px:  n={mt['n']}  "
          f"mean={mt['mean']:.1f}  p10={mt['p10']:.1f}  "
          f"p50={mt['p50']:.1f}  p90={mt['p90']:.1f}  "
          f"frac<10px={mt['frac_lt_10']*100:.2f}%")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label-a", required=True)
    ap.add_argument("--dir-a", type=Path, required=True)
    ap.add_argument("--label-b", required=True)
    ap.add_argument("--dir-b", type=Path, required=True)
    ap.add_argument("--views", nargs="+", default=list(VIEWS),
                    choices=list(VIEWS))
    args = ap.parse_args(argv)

    print(f"A = {args.label_a}  ({args.dir_a})")
    print(f"B = {args.label_b}  ({args.dir_b})")
    print()

    a_summ: dict[str, dict] = {}
    b_summ: dict[str, dict] = {}
    for view in args.views:
        print(f"=== {view} ===")
        for label, dir_, store in (
            (args.label_a, args.dir_a, a_summ),
            (args.label_b, args.dir_b, b_summ),
        ):
            slp = dir_ / f"{view}.slp"
            if not slp.exists():
                print(f"  [{label}] {view}: MISSING {slp}")
                continue
            s = summarise_slp(slp)
            store[view] = s
            print_summary(label, view, s)
        print()

    # Compact delta table
    print("=== summary delta (B - A) ===")
    print(f"  {'view':<12} {'A_inst':>10} {'B_inst':>10} {'A_3-rate':>9} "
          f"{'B_3-rate':>9} {'A_score':>9} {'B_score':>9}")
    for view in args.views:
        a = a_summ.get(view)
        b = b_summ.get(view)
        if not a or not b:
            continue
        a_3 = 100 * a["inst_count_hist"].get(3, 0) / max(1, a["n_frames_total"])
        b_3 = 100 * b["inst_count_hist"].get(3, 0) / max(1, b["n_frames_total"])
        print(f"  {view:<12} {a['n_instances']:>10} {b['n_instances']:>10} "
              f"{a_3:>8.1f}% {b_3:>8.1f}% "
              f"{a['mean_inst_score']:>9.3f} {b['mean_inst_score']:>9.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
