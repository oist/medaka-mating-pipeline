from __future__ import annotations

"""
Cross-round top-down accuracy comparison: how much did each retraining round
improve the medaka pose models?

Runs the TOP-DOWN pipeline (centroid + centered_instance) of several rounds on a
COMMON evaluation frame set (the round-7 merged labels, which span all three
experiments K_1_1 / K_1_2 / K_1_3) and reports the same detection / keypoint
metrics per round, split by experiment.

Why split by experiment
------------------------
- round-4 / round-5 were trained on K_1_2 ONLY.
- round-7 added K_1_1 + K_1_3 (rounds-6 annotation) + difficult K_1_2 frames.

So on K_1_2 the comparison is a regression check (did adding other-line data hurt
the original line?), and on K_1_1 / K_1_3 it is a generalization test (round-4/5
never saw those experiments -> out-of-sample for them, in-sample for round-7).

There is NO separate round-6 model: round-6 was an annotation round whose frames
were merged into round-7.

Usage
-----
    python -m scripts.sleap.compare_rounds_topdown --view top --device cuda \
        --limit 150 --out-json runs/compare_rounds_top.json
    python -m scripts.sleap.compare_rounds_topdown --view side --difficult-only
"""

import argparse
import json
from pathlib import Path

# Reuse the eval harness's inference + metric machinery.
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_topdown_vs_bottomup as E  # noqa: E402
from merge_cumulative_labels import (  # noqa: E402
    _experiment_of,
    _leaf_map,
    _norm_clip,
)
from split_pkg_by_view import _video_dataset_index  # noqa: E402

SLEAP_MODELS = Path("Z:/StephensU/medaka/sleap_models")

# Per round: (base round dir, run-name tag). n<N> suffix globbed off.
ROUND_SPECS = {
    "round4": (SLEAP_MODELS / "retrain_20260508", "round4_260520"),
    "round5": (SLEAP_MODELS / "retrain_20260611", "round5_260611"),
    "round7": (SLEAP_MODELS / "retrain_20260704", "round7_260704"),
}

# The GT / common eval frames come from round-7 (spans all three experiments).
GT_ROUND_DIR = SLEAP_MODELS / "retrain_20260704"

METRIC_KEYS = [
    ("detect_rate", "detect"),
    ("miss_rate", "miss"),
    ("merge_rate", "merge"),
    ("swap_rate", "swap"),
    ("mouth_err_p95", "mouth_p95"),
    ("tail_err_p95", "tail_p95"),
    ("oks_mean", "OKS"),
]


def resolve_topdown_dirs(base: Path, view: str, tag: str) -> tuple[Path, Path]:
    """Glob the centroid + centered_instance run dirs for one round/view."""
    runs = base / view / "runs"
    cents = sorted(runs.glob(f"{tag}.centroid_n*"))
    cis = sorted(runs.glob(f"{tag}.centered_instance_n*"))
    if not cents or not cis:
        raise SystemExit(
            f"missing top-down runs for {tag} under {runs} "
            f"(centroid={[c.name for c in cents]}, centered={[c.name for c in cis]})"
        )
    for d in (cents[-1], cis[-1]):
        if not (d / "best.ckpt").exists():
            raise SystemExit(f"{d} has no best.ckpt")
    return cents[-1], cis[-1]


def experiment_tags(gt_path: Path, labels) -> dict[int, str]:
    """Map each labeled-frame POSITION -> experiment (k_1_1 / k_1_2 / k_1_3)."""
    leaf = _leaf_map(gt_path)
    out: dict[int, str] = {}
    for i, lf in enumerate(labels.labeled_frames):
        ds = _video_dataset_index(lf.video)
        out[i] = _experiment_of(_norm_clip(leaf.get(ds) if ds is not None else None))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--view", required=True, choices=["top", "side"])
    ap.add_argument("--rounds", nargs="+", default=["round4", "round5", "round7"],
                    choices=list(ROUND_SPECS))
    ap.add_argument("--difficult-only", action="store_true")
    ap.add_argument("--close-contact-px", type=float, default=80.0)
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--match-gate-px", type=float, default=150.0)
    ap.add_argument("--max-instances", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args(argv)

    import sleap_io as sio

    view = args.view
    gt_path = GT_ROUND_DIR / view / "labels.pkg.slp"
    if not gt_path.exists():
        raise SystemExit(f"GT not found: {gt_path}")
    print(f"loading common GT: {gt_path}")
    labels = sio.load_slp(str(gt_path))
    node_names = [n.name for n in labels.skeletons[0].nodes]
    mouth_idx, tail_idx = node_names.index(E.MOUTH), node_names.index(E.TAIL)
    vid_index = {id(v): i for i, v in enumerate(labels.videos)}
    exp_of = experiment_tags(gt_path, labels)

    # Candidate frames (optionally close-contact only), deterministic subset.
    cand, keys = [], []
    for i, lf in enumerate(labels.labeled_frames):
        gt_pts = E._instances_as_arrays(lf.user_instances)
        if not gt_pts:
            continue
        if args.difficult_only and \
                E.min_pairwise_centroid_distance(gt_pts) > args.close_contact_px:
            continue
        cand.append(i)
        keys.append((vid_index.get(id(lf.video), -1), int(lf.frame_idx)))
    sel = E.select_eval_indices(keys, args.limit)
    eval_positions = [cand[i] for i in sel]
    exps = sorted({exp_of[p] for p in eval_positions})
    from collections import Counter
    exp_hist = Counter(exp_of[p] for p in eval_positions)
    print(f"candidate={len(cand)} -> eval subset={len(eval_positions)}  "
          f"by experiment: {dict(exp_hist)}")

    gt_subset_all = E.build_subset_labels(labels, eval_positions)
    exp_subsets = {
        exp: E.build_subset_labels(labels, [p for p in eval_positions
                                            if exp_of[p] == exp])
        for exp in exps
    }

    results: dict[str, dict] = {}
    for rnd in args.rounds:
        base, tag = ROUND_SPECS[rnd]
        cent, ci = resolve_topdown_dirs(base, view, tag)
        print(f"\n===== {rnd}: {cent.name} + {ci.name} =====")
        pred = E.run_strategy(
            strategy=f"top_down::{rnd}",
            model_paths=[cent, ci],
            subset_labels=E.build_subset_labels(labels, eval_positions),
            max_instances=args.max_instances,
            batch_size=args.batch_size,
            device=args.device,
        )
        overall = E.evaluate_strategy(
            strategy=rnd, gt_subset_labels=gt_subset_all, pred_labels=pred,
            mouth_idx=mouth_idx, tail_idx=tail_idx, gate=args.match_gate_px)
        per_exp = {
            exp: E.evaluate_strategy(
                strategy=f"{rnd}:{exp}", gt_subset_labels=exp_subsets[exp],
                pred_labels=pred, mouth_idx=mouth_idx, tail_idx=tail_idx,
                gate=args.match_gate_px)
            for exp in exps
        }
        results[rnd] = {"runs": [cent.name, ci.name],
                        "overall": overall, "per_experiment": per_exp}

    # ---- report ----
    def fmt(d):
        return "  ".join(f"{short}={d[key]:.3f}" for key, short in METRIC_KEYS)

    print("\n" + "=" * 70)
    print(f"CROSS-ROUND TOP-DOWN  view={view}  "
          f"difficult_only={args.difficult_only}  limit={args.limit}")
    print("=" * 70)
    print("\n[OVERALL]")
    for rnd in args.rounds:
        print(f"  {rnd:8s} {fmt(results[rnd]['overall'])}")
    for exp in exps:
        seen = "K_1_2 (all rounds saw it)" if "k_1_2" in exp else \
               f"{exp} (round-7 only saw it)"
        print(f"\n[{exp}]  {seen}")
        for rnd in args.rounds:
            print(f"  {rnd:8s} {fmt(results[rnd]['per_experiment'][exp])}")

    out = {"view": view, "difficult_only": args.difficult_only,
           "limit": args.limit, "n_eval_frames": len(eval_positions),
           "experiments": dict(exp_hist), "results": results}
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
