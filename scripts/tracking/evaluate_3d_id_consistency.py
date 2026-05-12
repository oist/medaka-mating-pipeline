from __future__ import annotations

"""
Evaluate 3D ID-tracking consistency on the bound `<exp>_3d_identified.csv`
(or any 3D-tracks CSV with columns `frame, fish_id (or final_fish_id), X, Y, Z`).

Mirrors `evaluate_id_consistency.py` but in 3D (mm/frame instead of px/frame),
so the same "wrong tracking gives spikes" signal applies to fish_id swaps in
the 3D pipeline.

Metrics per fish_id:
  - xyz_speed_p99 / p999  (mm/frame)
  - n_spikes_gt_T for T in {2.5, 5, 10, 20} mm/frame
    (medaka burst ~0.4 m/s = ~2.5 mm/frame at 166 fps; > 20 mm/frame is a swap)
  - spike_rate_gt_T
  - max_spike_free_run (frames, at T=10 mm)
  - spike_free_fish_count (T=10 mm)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))


SPIKE_THRESHOLDS_MM = (2.5, 5.0, 10.0, 20.0)
DEFAULT_SPIKE_T_MM = 10.0


def _fish_speed_series(tracks: pd.DataFrame, *, id_col: str
                       ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for fid, sub in tracks.groupby(id_col):
        sub = sub.sort_values("frame")
        f = sub["frame"].to_numpy()
        xyz = sub[["X", "Y", "Z"]].to_numpy(dtype=float)
        valid = np.isfinite(xyz).all(axis=1)
        f = f[valid]
        xyz = xyz[valid]
        if len(f) < 2:
            series[str(fid)] = (np.empty(0, dtype=int), np.empty(0, dtype=float))
            continue
        gap = np.diff(f)
        consec = gap == 1
        if not consec.any():
            series[str(fid)] = (np.empty(0, dtype=int), np.empty(0, dtype=float))
            continue
        dxyz = np.diff(xyz, axis=0)
        speeds = np.sqrt((dxyz ** 2).sum(axis=1))[consec]
        speed_frames = f[1:][consec]
        series[str(fid)] = (speed_frames, speeds)
    return series


def spike_metrics(tracks: pd.DataFrame, *, id_col: str) -> dict:
    series = _fish_speed_series(tracks, id_col=id_col)
    all_speeds = [sp for _, sp in series.values() if sp.size]
    if not all_speeds:
        out = {"xyz_speed_p99": float("nan"),
               "xyz_speed_p999": float("nan"),
               "max_spike_free_run": 0,
               "spike_free_fish_count": 0}
        for T in SPIKE_THRESHOLDS_MM:
            out[f"n_spikes_gt_{T}"] = 0
            out[f"spike_rate_gt_{T}"] = float("nan")
        return out

    sp = np.concatenate(all_speeds)
    out: dict = {
        "xyz_speed_p99": float(np.percentile(sp, 99)),
        "xyz_speed_p999": float(np.percentile(sp, 99.9)),
    }
    for T in SPIKE_THRESHOLDS_MM:
        n = int((sp > T).sum())
        out[f"n_spikes_gt_{T}"] = n
        out[f"spike_rate_gt_{T}"] = float(n / sp.size)

    T = DEFAULT_SPIKE_T_MM
    longest = 0
    clean = 0
    for _, (_, s) in series.items():
        if s.size == 0:
            continue
        is_clean = s <= T
        if is_clean.all():
            clean += 1
        run = 0
        for v in is_clean:
            run = run + 1 if v else 0
            if run > longest:
                longest = run
    out["max_spike_free_run"] = int(longest)
    out["spike_free_fish_count"] = int(clean)
    return out


def _save_speed_plot(tracks: pd.DataFrame, *, id_col: str,
                     title: str, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    series = _fish_speed_series(tracks, id_col=id_col)
    fig, ax = plt.subplots(figsize=(14, 4))
    colors = plt.cm.tab10.colors
    for i, (fid, (f, sp)) in enumerate(sorted(series.items())):
        if sp.size == 0:
            continue
        ax.plot(f, sp, lw=0.5, color=colors[i % len(colors)],
                label=f"{id_col}={fid} (n={sp.size})")
    ax.axhline(DEFAULT_SPIKE_T_MM, color="red", lw=0.8, ls="--",
               label=f"spike threshold = {DEFAULT_SPIKE_T_MM} mm/frame")
    ax.set_xlabel("frame")
    ax.set_ylabel("3D speed (mm / frame)")
    ax.set_title(title)
    ax.set_ylim(0, max(30, ax.get_ylim()[1]))
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate_csv(csv_path: Path, *, id_col: str | None = None,
                 plot_dir: Path | None = None) -> dict:
    tracks = pd.read_csv(csv_path)
    if id_col is None:
        id_col = "final_fish_id" if "final_fish_id" in tracks.columns else "fish_id"
    metrics = spike_metrics(tracks, id_col=id_col)
    if plot_dir is not None:
        _save_speed_plot(tracks, id_col=id_col,
                         title=csv_path.stem,
                         out_path=plot_dir / f"{csv_path.stem}__speed3d.png")
    metrics["id_col"] = id_col
    metrics["n_rows"] = len(tracks)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="3D ID-tracking spike-metric eval (mm/frame)."
    )
    p.add_argument("--csv", required=True, type=Path, nargs="+",
                   help="One or more *_3d_identified.csv (or *_3d.csv) paths.")
    p.add_argument("--id-col", default=None,
                   help="Column to group tracks by (default: final_fish_id if present, else fish_id).")
    p.add_argument("--plot-dir", type=Path, default=None,
                   help="Optional directory for per-csv speed plot PNGs.")
    p.add_argument("--out-csv", type=Path, default=None,
                   help="Optional summary CSV path (rows = input csvs).")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    rows = []
    for csv in args.csv:
        m = evaluate_csv(csv, id_col=args.id_col, plot_dir=args.plot_dir)
        m["csv"] = csv.name
        m["csv_path"] = str(csv)
        rows.append(m)
        print(f"[3d-eval] {csv.name}")
        for k, v in m.items():
            if k in {"csv", "csv_path"}:
                continue
            print(f"  {k}: {v}")
    if args.out_csv:
        df = pd.DataFrame(rows)
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out_csv, index=False)
        print(f"[3d-eval] wrote summary {args.out_csv}")


if __name__ == "__main__":
    main()
