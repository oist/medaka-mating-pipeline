from __future__ import annotations

"""
Visualise 3D trajectories from two reconstruction rounds for comparison.

Produces a single multi-panel PNG:

* Row 1 - top-down (X vs Y) scatter for round-A and round-B, side-by-side,
  colored by fish ID. Same axes so overlap/divergence is visually obvious.
* Row 2 - side view (X vs Z) for round-A and round-B.
* Row 3 - per-axis (X, Y, Z) time series with round-A dashed and round-B
  solid, for one matched fish (closest mean-distance pair between rounds).

Frames are downsampled to ~--max-points per fish to keep the figure legible
on 35-min recordings at 166 fps.

Usage::

    python -m scripts.reconstruction_3d.plot_rounds_3d \
        --label-a round3_3node_full --csv-a <path>/round3_3node_full_3d.csv \
        --label-b round4_260520     --csv-b <path>/round4_260520_3d.csv \
        --out <path>/round3_vs_round4_trajectories.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FISH_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def load_xyz(csv_path: Path) -> dict[str, np.ndarray]:
    """Return ``{fish_id: (n_frames, 3) np.ndarray}`` (NaN-padded)."""
    df = pd.read_csv(csv_path)
    n_frames = int(df["frame"].max()) + 1
    out: dict[str, np.ndarray] = {}
    for fid in sorted(df["fish_id"].unique()):
        arr = np.full((n_frames, 3), np.nan)
        sub = df[df["fish_id"] == fid].sort_values("frame")
        frames = sub["frame"].to_numpy()
        arr[frames, 0] = sub["X"].to_numpy()
        arr[frames, 1] = sub["Y"].to_numpy()
        arr[frames, 2] = sub["Z"].to_numpy()
        out[str(fid)] = arr
    return out


def match_ids(a: dict[str, np.ndarray],
              b: dict[str, np.ndarray]) -> dict[str, str]:
    """For each fish in A, return the fish in B with lowest mean per-frame distance."""
    out: dict[str, str] = {}
    used: set[str] = set()
    for fa, va in a.items():
        best_d = float("inf")
        best_fb: str | None = None
        for fb, vb in b.items():
            if fb in used:
                continue
            n = min(va.shape[0], vb.shape[0])
            d = np.linalg.norm(va[:n] - vb[:n], axis=1)
            d = d[~np.isnan(d)]
            mean_d = float(d.mean()) if d.size else float("inf")
            if mean_d < best_d:
                best_d = mean_d
                best_fb = fb
        if best_fb is not None:
            out[fa] = best_fb
            used.add(best_fb)
    return out


def downsample(arr: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Keep at most max_points rows; return (frame_idx, arr) of valid rows."""
    valid = ~np.isnan(arr[:, 0])
    idx = np.where(valid)[0]
    if idx.size > max_points:
        step = idx.size // max_points
        idx = idx[::step][:max_points]
    return idx, arr[idx]


def plot_xy(ax, data: dict[str, np.ndarray], title: str,
            max_points: int, xyz_lim: tuple[float, float]) -> None:
    for i, (fid, arr) in enumerate(data.items()):
        idx, xyz = downsample(arr, max_points)
        c = FISH_COLORS[i % len(FISH_COLORS)]
        ax.plot(xyz[:, 0], xyz[:, 1], color=c, lw=0.4, alpha=0.6, label=fid)
        # mark every Nth as scatter for visibility
        s = max(1, len(idx) // 200)
        ax.scatter(xyz[::s, 0], xyz[::s, 1], color=c, s=1.5, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
    ax.set_xlim(*xyz_lim); ax.set_ylim(*xyz_lim)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)


def plot_xz(ax, data: dict[str, np.ndarray], title: str,
            max_points: int, xyz_lim: tuple[float, float]) -> None:
    for i, (fid, arr) in enumerate(data.items()):
        idx, xyz = downsample(arr, max_points)
        c = FISH_COLORS[i % len(FISH_COLORS)]
        ax.plot(xyz[:, 0], xyz[:, 2], color=c, lw=0.4, alpha=0.6, label=fid)
        s = max(1, len(idx) // 200)
        ax.scatter(xyz[::s, 0], xyz[::s, 2], color=c, s=1.5, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Z (mm)")
    ax.set_xlim(*xyz_lim); ax.set_ylim(*xyz_lim)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)


def plot_axis_timeseries(ax, frames_a: np.ndarray, vals_a: np.ndarray,
                         frames_b: np.ndarray, vals_b: np.ndarray,
                         axis_name: str, label_a: str, label_b: str) -> None:
    ax.plot(frames_a, vals_a, color="#888", lw=0.6, ls="--", label=label_a, alpha=0.8)
    ax.plot(frames_b, vals_b, color="#d62728", lw=0.6, label=label_b, alpha=0.9)
    ax.set_ylabel(f"{axis_name} (mm)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label-a", required=True)
    ap.add_argument("--csv-a", type=Path, required=True)
    ap.add_argument("--label-b", required=True)
    ap.add_argument("--csv-b", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-points", type=int, default=4000,
                    help="Max samples per fish per panel (default 4000).")
    ap.add_argument("--xyz-lim", nargs=2, type=float, default=(0.0, 220.0),
                    metavar=("LOW", "HIGH"))
    args = ap.parse_args(argv)

    print(f"Loading {args.csv_a} ...")
    a = load_xyz(args.csv_a)
    print(f"Loading {args.csv_b} ...")
    b = load_xyz(args.csv_b)
    print(f"  A: {list(a.keys())}; B: {list(b.keys())}")

    matches = match_ids(a, b)
    print(f"  ID matching A->B: {matches}")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14, 14))
    gs = fig.add_gridspec(4, 2, height_ratios=[2.2, 2.2, 1, 1, ][:3] + [1])

    ax_xy_a = fig.add_subplot(gs[0, 0])
    ax_xy_b = fig.add_subplot(gs[0, 1])
    plot_xy(ax_xy_a, a, f"{args.label_a} (top-down X-Y)", args.max_points, tuple(args.xyz_lim))
    plot_xy(ax_xy_b, b, f"{args.label_b} (top-down X-Y)", args.max_points, tuple(args.xyz_lim))

    ax_xz_a = fig.add_subplot(gs[1, 0])
    ax_xz_b = fig.add_subplot(gs[1, 1])
    plot_xz(ax_xz_a, a, f"{args.label_a} (side X-Z)", args.max_points, tuple(args.xyz_lim))
    plot_xz(ax_xz_b, b, f"{args.label_b} (side X-Z)", args.max_points, tuple(args.xyz_lim))

    # Pick the best-matched fish pair for the time-series row.
    fa = next(iter(matches.keys()))
    fb = matches[fa]
    print(f"  time-series shows: A[{fa}] vs B[{fb}]")
    va, vb = a[fa], b[fb]
    n = min(va.shape[0], vb.shape[0])
    va, vb = va[:n], vb[:n]
    valid = ~np.isnan(va[:, 0]) & ~np.isnan(vb[:, 0])
    idx = np.where(valid)[0]
    if idx.size > args.max_points * 2:
        step = idx.size // (args.max_points * 2)
        idx = idx[::step]

    ax_ts = fig.add_subplot(gs[2, :])
    plot_axis_timeseries(ax_ts, idx, va[idx, 0], idx, vb[idx, 0], "X",
                         f"{args.label_a}[{fa}]", f"{args.label_b}[{fb}]")
    ax_ts.set_title(f"Time series (matched IDs: A[{fa}] vs B[{fb}])")

    # We need a 4th row but the gridspec was 4 rows for 3 panels; collapse:
    # use a single ax with multiple lines (X, Y, Z stacked).
    ax_ts.set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {args.out} ({args.out.stat().st_size/1e6:.1f} MB)")

    # Additional figure: stacked X/Y/Z time series for the matched pair.
    ts_out = args.out.with_name(args.out.stem + "_timeseries.png")
    fig2, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    for ax, axis_name, ci in zip(axes, ("X", "Y", "Z"), (0, 1, 2)):
        plot_axis_timeseries(ax, idx, va[idx, ci], idx, vb[idx, ci], axis_name,
                             f"{args.label_a}[{fa}]", f"{args.label_b}[{fb}]")
    axes[-1].set_xlabel("frame")
    fig2.suptitle(f"Per-axis trajectory: {args.label_a}[{fa}] vs {args.label_b}[{fb}]")
    fig2.tight_layout()
    fig2.savefig(ts_out, dpi=120, bbox_inches="tight")
    plt.close(fig2)
    print(f"  wrote {ts_out} ({ts_out.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
