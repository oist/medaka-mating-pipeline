"""Per-ID body-length-over-time (side views) + female identification + crossings.

The female is the consistently-largest fish and is distinguishable by eye in the
side view. For one experiment this maps each side-view SLEAP instance to its
unified 3D identity (nearest reprojection of the 3D fish into that view), takes
the per-ID mouth-tail length (max across the two orthogonal side cams to beat
foreshortening), smooths it, and plots the three IDs over time:

* The female ID should sit clearly above the two males the whole recording.
* An ID swap shows as the lines crossing / dipping at a specific frame -> the
  x-axis tells you exactly where to look (and feeds the crossing montage).

Also detects female<->male close approaches (3D) and writes a crossings CSV used
by ``render_crossing_montage.py``.

Outputs (next to the 3D csv): ``<tag>_id_size_timeline.png``,
``<tag>_crossings.csv``, and a printed summary (female ID, per-ID median length).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.reconstruction_3d.calibration import Calibration  # noqa: E402
from scripts.analysis.render_predictions_identity_video import (  # noqa: E402
    load_view_predictions, assign_identity, VIEW_CAM_IDX, VIEW_SLP_NAME, ID_COLORS,
)

SIDE_VIEWS = ["side1", "side2"]
SMOOTH_WIN = 166          # ~1 s rolling median
CROSS_MM = 18.0           # female-male 3D distance defining a close approach
CROSS_MERGE_GAP = 166     # merge events separated by < ~1 s
FPS = 166.0


def per_id_length(df, cal, preds, id_col, max_dist=60.0):
    ids = sorted(int(v) for v in df[id_col].dropna().unique())
    frames = df["frame"].to_numpy()
    fids = df[id_col].to_numpy()
    xyz = df[["X", "Y", "Z"]].to_numpy(float)
    finite = np.all(np.isfinite(xyz), axis=1)
    uv = np.full((len(df), 3, 2), np.nan)
    if finite.any():
        uv[finite] = cal.compute_imCoords_from_XYZ(xyz[finite])

    reproj = defaultdict(lambda: {v: [] for v in SIDE_VIEWS})
    for i in range(len(df)):
        if not finite[i] or not np.isfinite(fids[i]):
            continue
        fid, f = int(fids[i]), int(frames[i])
        for v in SIDE_VIEWS:
            x, y = uv[i, VIEW_CAM_IDX[v], 0], uv[i, VIEW_CAM_IDX[v], 1]
            if np.isfinite(x) and np.isfinite(y):
                reproj[f][v].append((fid, x, y))

    allframes = np.array(sorted(set(int(f) for f in frames)))
    fidx = {f: i for i, f in enumerate(allframes)}
    series = {fid: np.full(len(allframes), np.nan) for fid in ids}
    for v in SIDE_VIEWS:
        for f, insts in preds[v].items():
            j = fidx.get(int(f))
            if j is None:
                continue
            rp = reproj[int(f)][v]
            for inst in insts:
                if inst.shape[0] < 2:
                    continue
                m, t = inst[0], inst[1]
                if not (np.all(np.isfinite(m)) and np.all(np.isfinite(t))):
                    continue
                aid = assign_identity(np.array([m, t]).mean(axis=0), rp, max_dist)
                if aid is None:
                    continue
                L = float(np.hypot(*(m - t)))
                cur = series[aid][j]
                series[aid][j] = L if not np.isfinite(cur) else max(cur, L)
    return ids, allframes, series


def detect_crossings(df, female_id, id_col):
    """Frames where the female is within CROSS_MM of a male (3D), as merged events."""
    piv = df.pivot_table(index="frame", columns=id_col, values=["X", "Y", "Z"])
    frames = piv.index.to_numpy()
    males = [c for c in sorted(df[id_col].dropna().unique()) if int(c) != female_id]
    fem = np.stack([piv[("X", female_id)], piv[("Y", female_id)], piv[("Z", female_id)]], axis=1)
    close = np.zeros(len(frames), bool)
    for mle in males:
        mp = np.stack([piv[("X", mle)], piv[("Y", mle)], piv[("Z", mle)]], axis=1)
        d = np.linalg.norm(fem - mp, axis=1)
        close |= (d < CROSS_MM)
    # merge contiguous (with small-gap bridging) into events -> center frames
    events, run_start, last = [], None, None
    for k, f in enumerate(frames):
        if close[k]:
            if run_start is None:
                run_start = f
            last = f
        elif run_start is not None and f - last > CROSS_MERGE_GAP:
            events.append((run_start, last)); run_start = None
    if run_start is not None:
        events.append((run_start, last))
    return events


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv-3d", type=Path, required=True)
    ap.add_argument("--slp-dir", type=Path, required=True)
    ap.add_argument("--model-folder", type=Path, required=True)
    ap.add_argument("--out-prefix", type=Path, required=True,
                    help="Output path prefix; writes <prefix>_id_size_timeline.png "
                         "and <prefix>_crossings.csv.")
    ap.add_argument("--female-id", type=int, default=None,
                    help="Override the auto-detected female ID.")
    args = ap.parse_args()

    df = pd.read_csv(args.csv_3d, usecols=lambda c: c in
                     ("frame", "final_fish_id", "fish_id", "X", "Y", "Z"))
    id_col = "final_fish_id" if "final_fish_id" in df.columns else "fish_id"
    cal = Calibration(args.model_folder)
    preds = {v: load_view_predictions(args.slp_dir / VIEW_SLP_NAME[v]) for v in SIDE_VIEWS}

    ids, frames, series = per_id_length(df, cal, preds, id_col)
    medians = {fid: float(np.nanmedian(series[fid])) for fid in ids}
    female = args.female_id if args.female_id is not None else max(medians, key=medians.get)

    print(f"per-ID median side body-length (px): "
          + "  ".join(f"ID{fid}={medians[fid]:.1f}" for fid in ids))
    print(f"=> female (largest) = ID {female}")

    # smoothed timeline plot
    fig, ax = plt.subplots(figsize=(16, 5))
    t = frames / FPS
    for fid in ids:
        s = pd.Series(series[fid]).rolling(SMOOTH_WIN, min_periods=20, center=True).median()
        c = np.array(ID_COLORS[fid % len(ID_COLORS)][::-1]) / 255.0  # BGR->RGB
        lab = f"ID {fid}" + ("  (FEMALE)" if fid == female else "")
        ax.plot(t, s, color=c, lw=1.5 if fid == female else 1.0, label=lab,
                alpha=0.9 if fid == female else 0.7)
    ax.set_xlabel("time (s)"); ax.set_ylabel("body length, side view (px)")
    ax.set_title(f"Per-ID body length over time  ({args.out_prefix.name})\n"
                 "female should stay highest; crossings/dips = candidate ID swaps",
                 fontsize=11)
    ax.legend(loc="upper right"); ax.grid(alpha=0.3)
    png = args.out_prefix.parent / f"{args.out_prefix.name}_id_size_timeline.png"
    fig.tight_layout(); fig.savefig(png, dpi=120)
    print("wrote", png)

    events = detect_crossings(df, female, id_col)
    cross_csv = args.out_prefix.parent / f"{args.out_prefix.name}_crossings.csv"
    pd.DataFrame([{"start": s, "end": e, "center": (s + e) // 2,
                   "dur_frames": e - s + 1} for s, e in events]).to_csv(cross_csv, index=False)
    print(f"detected {len(events)} female-male crossing events (<{CROSS_MM:.0f} mm) -> {cross_csv}")


if __name__ == "__main__":
    main()
