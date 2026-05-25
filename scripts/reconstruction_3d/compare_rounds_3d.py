from __future__ import annotations

"""
Compare round-N vs round-(N-1) 3D reconstruction outputs.

For two ``<tag>_3d.csv`` files produced by ``reconstruct_from_slp``, report
metrics that distinguish "did the new model give better 3D tracking":

* Coverage: per-fish fill rate, fraction of frames with all-3-fish valid.
* Track break stats: number of NaN runs per fish, longest gap, mean gap.
* Motion smoothness: per-frame velocity p50 / p95, fraction of implausible
  jumps (> --vel-threshold mm/frame, default 30).
* If ``<tag>_best_pre_link.json`` exists, summarise triangulation residual.

Usage::

    python -m scripts.reconstruction_3d.compare_rounds_3d \
        --label-a round3_3node_full --csv-a <path>/round3_3node_full_3d.csv \
        --label-b round4_260520     --csv-b <path>/round4_260520_3d.csv
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def runs_of_nan(mask_nan: np.ndarray) -> list[int]:
    """Lengths of consecutive-True runs in a boolean array."""
    runs: list[int] = []
    run = 0
    for v in mask_nan:
        if v:
            run += 1
        else:
            if run:
                runs.append(run)
            run = 0
    if run:
        runs.append(run)
    return runs


def summarise_csv(csv_path: Path, vel_threshold_mm: float = 30.0) -> dict:
    df = pd.read_csv(csv_path)
    n_frames = int(df["frame"].max()) + 1
    fishes = sorted(df["fish_id"].unique())
    n_fish = len(fishes)

    # Pivot to (n_frames, n_fish, 3) tensor.
    arr = np.full((n_frames, n_fish, 3), np.nan)
    for fi, fid in enumerate(fishes):
        sub = df[df["fish_id"] == fid].sort_values("frame")
        # Frames may have gaps; reindex.
        frames = sub["frame"].to_numpy()
        arr[frames, fi, 0] = sub["X"].to_numpy()
        arr[frames, fi, 1] = sub["Y"].to_numpy()
        arr[frames, fi, 2] = sub["Z"].to_numpy()

    valid = ~np.isnan(arr[:, :, 0])  # (n_frames, n_fish)
    all_valid = np.all(valid, axis=1)
    any_valid = np.any(valid, axis=1)
    fill_rate_per_fish = {f: float(valid[:, i].mean()) for i, f in enumerate(fishes)}

    # Gap statistics per fish.
    gap_stats = {}
    for i, fid in enumerate(fishes):
        nan_mask = ~valid[:, i]
        rs = runs_of_nan(nan_mask)
        gap_stats[fid] = {
            "n_gaps": len(rs),
            "max_gap": int(max(rs)) if rs else 0,
            "mean_gap": float(np.mean(rs)) if rs else 0.0,
            "median_gap": float(np.median(rs)) if rs else 0.0,
            "total_nan_frames": int(sum(rs)),
        }

    # Velocity.
    diff = np.diff(arr, axis=0)
    speed = np.linalg.norm(diff, axis=2)  # (n_frames-1, n_fish)
    speed = speed[~np.isnan(speed)]
    if speed.size > 0:
        vel_p50 = float(np.percentile(speed, 50))
        vel_p95 = float(np.percentile(speed, 95))
        vel_max = float(speed.max())
        frac_jumps = float(np.mean(speed > vel_threshold_mm))
    else:
        vel_p50 = vel_p95 = vel_max = frac_jumps = float("nan")

    # Spatial extent (sanity).
    finite = arr[~np.isnan(arr).any(axis=2)]
    if finite.size:
        xyz_min = finite.min(axis=0).tolist()
        xyz_max = finite.max(axis=0).tolist()
    else:
        xyz_min = xyz_max = None

    return {
        "n_frames": n_frames,
        "n_fish": n_fish,
        "fishes": fishes,
        "fill_rate_per_fish": fill_rate_per_fish,
        "all_3_valid_rate": float(all_valid.mean()),
        "any_valid_rate": float(any_valid.mean()),
        "total_valid_cells": int(valid.sum()),
        "total_cells": int(valid.size),
        "gap_stats": gap_stats,
        "vel_p50": vel_p50,
        "vel_p95": vel_p95,
        "vel_max": vel_max,
        "frac_jumps_over_threshold": frac_jumps,
        "vel_threshold_mm": vel_threshold_mm,
        "xyz_min": xyz_min,
        "xyz_max": xyz_max,
    }


def summarise_pre_link(json_path: Path) -> dict | None:
    if not json_path.exists():
        return None
    with json_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    errs: list[float] = []
    for frame_info in data["frames"].values():
        for v in frame_info["errors"].values():
            if v is not None and not math.isnan(float(v)):
                errs.append(float(v))
    if not errs:
        return {"n": 0}
    a = np.asarray(errs)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p90": float(np.percentile(a, 90)),
        "p99": float(np.percentile(a, 99)),
        "frac_over_5000": float(np.mean(a > 5000)),
    }


def print_summary(label: str, s: dict) -> None:
    print(f"=== {label} ===")
    print(f"  frames           : {s['n_frames']}")
    print(f"  fishes           : {s['fishes']}")
    print(f"  fill rate / fish : "
          + "  ".join(f"{f}={r*100:.2f}%" for f, r in s["fill_rate_per_fish"].items()))
    print(f"  all-3-valid rate : {s['all_3_valid_rate']*100:.2f}%")
    print(f"  any-valid rate   : {s['any_valid_rate']*100:.2f}%")
    print(f"  total valid      : {s['total_valid_cells']}/{s['total_cells']} "
          f"({100*s['total_valid_cells']/s['total_cells']:.2f}%)")
    print(f"  velocity (mm/fr) : p50={s['vel_p50']:.3f}  p95={s['vel_p95']:.3f}  "
          f"max={s['vel_max']:.1f}  frac>{s['vel_threshold_mm']:.0f}={s['frac_jumps_over_threshold']*100:.3f}%")
    print(f"  xyz min/max      : min={s['xyz_min']}  max={s['xyz_max']}")
    print(f"  gap stats:")
    for fid, g in s["gap_stats"].items():
        print(f"    {fid}: n_gaps={g['n_gaps']:>6}  "
              f"max_gap={g['max_gap']:>7}  mean_gap={g['mean_gap']:.1f}  "
              f"total_nan={g['total_nan_frames']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label-a", required=True)
    ap.add_argument("--csv-a", type=Path, required=True)
    ap.add_argument("--label-b", required=True)
    ap.add_argument("--csv-b", type=Path, required=True)
    ap.add_argument("--vel-threshold", type=float, default=30.0,
                    help="mm/frame above which to count an implausible jump.")
    args = ap.parse_args(argv)

    a = summarise_csv(args.csv_a, vel_threshold_mm=args.vel_threshold)
    b = summarise_csv(args.csv_b, vel_threshold_mm=args.vel_threshold)
    print_summary(args.label_a, a)
    print()
    print_summary(args.label_b, b)
    print()

    # Pre-link residuals.
    for label, csv_p in ((args.label_a, args.csv_a), (args.label_b, args.csv_b)):
        pre = csv_p.with_name(csv_p.name.replace("_3d.csv", "_best_pre_link.json"))
        pls = summarise_pre_link(pre)
        if pls is None:
            continue
        print(f"=== {label} triangulation residual (pre-link) ===")
        if pls.get("n") == 0:
            print("  (no residuals)")
        else:
            print(f"  n={pls['n']}  mean={pls['mean']:.1f}  median={pls['median']:.1f}  "
                  f"p90={pls['p90']:.1f}  p99={pls['p99']:.1f}  "
                  f"frac>5000={pls['frac_over_5000']*100:.2f}%")
        print()

    # Compact delta line.
    print("=== summary delta (B - A) ===")
    print(f"  {'metric':<30} {'A':>14} {'B':>14}   delta")
    print(f"  {'all-3-valid %':<30} {a['all_3_valid_rate']*100:>13.2f}% "
          f"{b['all_3_valid_rate']*100:>13.2f}% "
          f"  {(b['all_3_valid_rate']-a['all_3_valid_rate'])*100:+.2f} pp")
    print(f"  {'total valid %':<30} "
          f"{100*a['total_valid_cells']/a['total_cells']:>13.2f}% "
          f"{100*b['total_valid_cells']/b['total_cells']:>13.2f}% "
          f"  {100*(b['total_valid_cells']/b['total_cells']-a['total_valid_cells']/a['total_cells']):+.2f} pp")
    print(f"  {'velocity p95 (mm/fr)':<30} {a['vel_p95']:>14.3f} {b['vel_p95']:>14.3f}   "
          f"{b['vel_p95']-a['vel_p95']:+.3f}")
    print(f"  {'frac jumps > 30 mm/fr (%)':<30} "
          f"{a['frac_jumps_over_threshold']*100:>13.3f}% "
          f"{b['frac_jumps_over_threshold']*100:>13.3f}% "
          f"  {(b['frac_jumps_over_threshold']-a['frac_jumps_over_threshold'])*100:+.3f} pp")
    total_gaps_a = sum(g["n_gaps"] for g in a["gap_stats"].values())
    total_gaps_b = sum(g["n_gaps"] for g in b["gap_stats"].values())
    print(f"  {'total gaps across fish':<30} {total_gaps_a:>14} {total_gaps_b:>14}   "
          f"{total_gaps_b-total_gaps_a:+d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
