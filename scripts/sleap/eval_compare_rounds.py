from __future__ import annotations

"""
Validate and compare trained SLEAP-NN models across retraining rounds.

For each (round, view, kind) model this runs ``sleap_nn.cli eval`` on the
held-out validation split that training saved next to the checkpoint
(``labels_gt.val.0.slp`` = ground truth, ``labels_pr.val.0.slp`` = the model's
predictions on it, generated from the BEST checkpoint), collects the OKS / mAP /
distance / PCK metrics, prints a comparison table, and renders a grouped-bar
figure (round-A vs round-B per model type).

The val split differs per round (each round draws its own random 10% holdout
from its own label set), so this is an each-model-on-its-own-holdout
generalisation comparison, NOT an identical-data head-to-head. Distances are in
full-image pixels and mOKS uses the GT instance scale, so both are comparable
across rounds (same fish, same px scale). For centered_instance, ``dist.p99`` is
the mouth<->tail node-swap signal (a swap ~= one body length of error).

Usage
-----
    python -m scripts.sleap.eval_compare_rounds \
        --out-dir Z:/StephensU/medaka/sleap_models/eval_round4_vs_round5

Edit the ``RUNS`` table below to point at the rounds you want to compare.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SLEAP_MODELS = Path("Z:/StephensU/medaka/sleap_models")

# (round_label, view, kind, run_dir relative to SLEAP_MODELS)
RUNS = [
    ("round4", "top",  "centroid",          "retrain_20260508/top/runs/round4_260520.centroid_n667"),
    ("round4", "top",  "centered_instance", "retrain_20260508/top/runs/round4_260520.centered_instance_n667"),
    ("round4", "side", "centroid",          "retrain_20260508/side/runs/round4_260520.centroid_n1333"),
    ("round4", "side", "centered_instance", "retrain_20260508/side/runs/round4_260520.centered_instance_n1333"),
    ("round5", "top",  "centroid",          "retrain_20260611/top/runs/round5_260611.centroid_n900"),
    ("round5", "top",  "centered_instance", "retrain_20260611/top/runs/round5_260611.centered_instance_n900"),
    ("round5", "side", "centroid",          "retrain_20260611/side/runs/round5_260611.centroid_n1800"),
    ("round5", "side", "centered_instance", "retrain_20260611/side/runs/round5_260611.centered_instance_n1800"),
]

ROUNDS = ("round4", "round5")
MODEL_TYPES = [("top", "centroid"), ("top", "centered_instance"),
               ("side", "centroid"), ("side", "centered_instance")]


def run_eval(run_dir: Path, out_npz: Path) -> dict | None:
    """Run ``sleap_nn.cli eval`` on a run's saved val GT/PR pair; load metrics."""
    gt = run_dir / "labels_gt.val.0.slp"
    pr = run_dir / "labels_pr.val.0.slp"
    if not (gt.exists() and pr.exists()):
        print(f"  MISSING val pair in {run_dir}")
        return None
    # Default OKS params: in the CLI, match_threshold is an OKS value (0-1);
    # leaving it at the default is what reproduces the training-time metrics.
    cmd = [sys.executable, "-m", "sleap_nn.cli", "eval",
           "-g", str(gt), "-p", str(pr), "-s", str(out_npz)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not out_npz.exists():
        print(f"  eval failed for {run_dir.name}:\n{proc.stderr[-500:]}")
        return None
    d = np.load(out_npz, allow_pickle=True)["metrics"].item()
    dm = d["distance_metrics"]
    return {
        "mOKS": float(d["mOKS"]["mOKS"]),
        "mAP": float(d["voc_metrics"]["oks_voc.mAP"]),
        "mAR": float(d["voc_metrics"]["oks_voc.mAR"]),
        "dist_avg": float(dm["avg"]),
        "dist_p50": float(dm["p50"]),
        "dist_p90": float(dm["p90"]),
        "dist_p95": float(dm["p95"]),
        "dist_p99": float(dm["p99"]),
        "PCK5": float(d["pck_metrics"]["PCK@5"]),
        "PCK10": float(d["pck_metrics"]["PCK@10"]),
        "n_dist": int(np.size(dm["dists"])),
    }


def collect(out_dir: Path) -> dict[tuple[str, str, str], dict]:
    results: dict[tuple[str, str, str], dict] = {}
    for rnd, view, kind, rel in RUNS:
        run_dir = SLEAP_MODELS / rel
        key = (rnd, view, kind)
        print(f"[eval] {rnd} {view}/{kind}")
        m = run_eval(run_dir, out_dir / f"eval_{rnd}_{view}_{kind}.npz")
        if m is not None:
            results[key] = m
            print(f"   mOKS={m['mOKS']:.3f} mAP={m['mAP']:.3f} "
                  f"dist(avg/p90/p99)={m['dist_avg']:.2f}/{m['dist_p90']:.2f}/"
                  f"{m['dist_p99']:.2f}px PCK@5={m['PCK5']:.3f} n={m['n_dist']}")
    return results


def print_table(results: dict) -> list[str]:
    lines = []
    hdr = (f"{'model':<26}{'round':<8}{'mOKS':>7}{'mAP':>7}{'dist_avg':>9}"
           f"{'p90':>8}{'p99':>8}{'PCK@5':>8}{'n':>6}")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for view, kind in MODEL_TYPES:
        for rnd in ROUNDS:
            m = results.get((rnd, view, kind))
            if not m:
                continue
            lines.append(
                f"{view+'/'+kind:<26}{rnd:<8}{m['mOKS']:>7.3f}{m['mAP']:>7.3f}"
                f"{m['dist_avg']:>9.2f}{m['dist_p90']:>8.2f}{m['dist_p99']:>8.2f}"
                f"{m['PCK5']:>8.3f}{m['n_dist']:>6}")
        lines.append("")
    return lines


def _grouped_bars(ax, results, metric, title, ylabel, logy=False):
    labels = [f"{v}\n{k.replace('centered_instance','c-inst')}"
              for v, k in MODEL_TYPES]
    x = np.arange(len(MODEL_TYPES))
    w = 0.38
    colors = {"round4": "#9aa0a6", "round5": "#1a73e8"}
    for i, rnd in enumerate(ROUNDS):
        vals = [results.get((rnd, v, k), {}).get(metric, np.nan)
                for v, k in MODEL_TYPES]
        bars = ax.bar(x + (i - 0.5) * w, vals, w, label=rnd, color=colors[rnd])
        for b, val in zip(bars, vals):
            if np.isfinite(val):
                ax.annotate(f"{val:.2f}" if val >= 0.1 or val == 0 else f"{val:.3f}",
                            (b.get_x() + b.get_width() / 2, b.get_height()),
                            ha="center", va="bottom", fontsize=7,
                            xytext=(0, 1), textcoords="offset points")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=8)
    if logy:
        ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def _ci_percentiles(ax, results):
    """centered_instance distance percentiles (the node-swap signal)."""
    ci_types = [("top", "centered_instance"), ("side", "centered_instance")]
    percs = ["dist_p50", "dist_p90", "dist_p99"]
    plabels = ["p50", "p90", "p99"]
    groups = []
    vals_by = {"round4": [], "round5": []}
    for v, k in ci_types:
        for pi, p in enumerate(percs):
            groups.append(f"{v}\n{plabels[pi]}")
            for rnd in ROUNDS:
                vals_by[rnd].append(results.get((rnd, v, k), {}).get(p, np.nan))
    x = np.arange(len(groups))
    w = 0.38
    colors = {"round4": "#9aa0a6", "round5": "#1a73e8"}
    for i, rnd in enumerate(ROUNDS):
        ax.bar(x + (i - 0.5) * w, vals_by[rnd], w, label=rnd, color=colors[rnd])
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=8)
    ax.set_title("centered_instance node-distance percentiles (px)\n"
                 "p99 = mouth/tail swap signal", fontsize=10, fontweight="bold")
    ax.set_ylabel("pixels", fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def make_figure(results: dict, out_png: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    _grouped_bars(axes[0, 0], results, "mOKS",
                  "mean OKS (higher = better)", "mOKS")
    axes[0, 0].set_ylim(0, 1.05)
    _grouped_bars(axes[0, 1], results, "mAP",
                  "mAP (OKS VOC, higher = better)", "mAP")
    axes[0, 1].set_ylim(0, 1.05)
    _grouped_bars(axes[1, 0], results, "dist_avg",
                  "mean keypoint distance (lower = better)",
                  "px (log)", logy=True)
    _ci_percentiles(axes[1, 1], results)
    fig.suptitle("SLEAP-NN held-out eval: round4 vs round5  "
                 "(each model on its own 10% val split)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=130)
    print(f"\nwrote figure -> {out_png}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path,
                    default=SLEAP_MODELS / "eval_round4_vs_round5")
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = collect(args.out_dir)
    lines = print_table(results)
    print()
    for ln in lines:
        print(ln)
    table_path = args.out_dir / "metrics_table.txt"
    table_path.write_text("\n".join(lines), encoding="utf-8")
    make_figure(results, args.out_dir / "round4_vs_round5_eval.png")
    print(f"wrote table  -> {table_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
