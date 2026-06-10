from __future__ import annotations

"""Build time-series diagnostics for tuning early 2D tracking motion gates."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))


def _wrap_pi(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def _nearest_neighbor_distance(df: pd.DataFrame) -> pd.Series:
    nearest = np.full(len(df), np.nan, dtype=float)
    positions = {idx: pos for idx, pos in zip(df.index, df[["x", "y"]].to_numpy(dtype=float))}
    for _, sub in df.groupby("frame", sort=False):
        if len(sub) < 2:
            continue
        pts = sub[["x", "y"]].to_numpy(dtype=float)
        dist = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
        np.fill_diagonal(dist, np.inf)
        nn = np.min(dist, axis=1)
        for row_idx, value in zip(sub.index, nn):
            if row_idx in positions:
                nearest[df.index.get_loc(row_idx)] = float(value)
    return pd.Series(nearest, index=df.index)


def compute_motion_timeseries(
    tracks: pd.DataFrame,
    *,
    close_contact_distance_px: float = 35.0,
    windows: tuple[int, ...] = (15, 45, 90),
) -> pd.DataFrame:
    required = {"frame", "track_id", "x", "y"}
    missing = required - set(tracks.columns)
    if missing:
        raise ValueError(f"tracks missing columns: {sorted(missing)}")

    df = tracks.copy()
    df = df.sort_values(["track_id", "frame"]).reset_index(drop=True)
    df["nearest_neighbor_distance_px"] = _nearest_neighbor_distance(df)
    df["close_contact"] = df["nearest_neighbor_distance_px"] <= float(close_contact_distance_px)

    pieces: list[pd.DataFrame] = []
    for _, sub in df.groupby("track_id", sort=False):
        sub = sub.sort_values("frame").copy()
        frame = sub["frame"].to_numpy(dtype=float)
        x = sub["x"].to_numpy(dtype=float)
        y = sub["y"].to_numpy(dtype=float)

        dt = np.r_[np.nan, np.diff(frame)]
        dx = np.r_[np.nan, np.diff(x)]
        dy = np.r_[np.nan, np.diff(y)]
        safe_dt = np.where((dt > 0) & np.isfinite(dt), dt, np.nan)
        vx = dx / safe_dt
        vy = dy / safe_dt
        speed = np.hypot(vx, vy)
        heading = np.arctan2(vy, vx)

        heading_change = np.r_[np.nan, _wrap_pi(np.diff(heading))]
        angular_velocity = np.abs(heading_change) / safe_dt

        prev_vx = np.r_[np.nan, vx[:-1]]
        prev_vy = np.r_[np.nan, vy[:-1]]
        prev_speed = np.hypot(prev_vx, prev_vy)
        dvx = vx - prev_vx
        dvy = vy - prev_vy
        prev_unit_x = np.divide(prev_vx, prev_speed, out=np.full_like(prev_vx, np.nan), where=prev_speed > 1e-6)
        prev_unit_y = np.divide(prev_vy, prev_speed, out=np.full_like(prev_vy, np.nan), where=prev_speed > 1e-6)
        tangent_accel = (dvx * prev_unit_x + dvy * prev_unit_y) / safe_dt
        lateral_accel = np.abs(dvx * prev_unit_y - dvy * prev_unit_x) / safe_dt

        sub["frame_gap"] = dt
        sub["vx_px_per_frame"] = vx
        sub["vy_px_per_frame"] = vy
        sub["speed_px_per_frame"] = speed
        sub["heading_rad"] = heading
        sub["heading_change_rad"] = np.abs(heading_change)
        sub["angular_velocity_rad_per_frame"] = angular_velocity
        sub["tangent_accel_px_per_frame2"] = tangent_accel
        sub["lateral_accel_px_per_frame2"] = lateral_accel
        sub["continuous_step"] = sub["frame_gap"] == 1

        for win in windows:
            roll = sub.rolling(window=int(win), min_periods=max(3, min(5, int(win))))
            sub[f"bin{win}_speed_median"] = roll["speed_px_per_frame"].median()
            sub[f"bin{win}_speed_p90"] = roll["speed_px_per_frame"].quantile(0.90)
            sub[f"bin{win}_angular_median"] = roll["angular_velocity_rad_per_frame"].median()
            sub[f"bin{win}_tangent_accel_median"] = roll["tangent_accel_px_per_frame2"].median()
            sub[f"bin{win}_lateral_accel_median"] = roll["lateral_accel_px_per_frame2"].median()
            sub[f"bin{win}_continuity_fraction"] = roll["continuous_step"].mean()
            sub[f"bin{win}_close_fraction"] = roll["close_contact"].mean()

        pieces.append(sub)

    out = pd.concat(pieces, ignore_index=True) if pieces else df
    out = _add_weak_bands(out)
    return out


def _add_weak_bands(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    clean = out.loc[
        (~out["close_contact"].fillna(False))
        & out["speed_px_per_frame"].replace([np.inf, -np.inf], np.nan).notna(),
        "speed_px_per_frame",
    ]
    if clean.empty:
        out["speed_band"] = "unknown"
        out["speed_upper_q95"] = np.nan
        out["speed_upper_q99"] = np.nan
        out["speed_upper_q999"] = np.nan
        return out

    q10, q50, q95, q99, q999 = np.nanpercentile(clean.to_numpy(dtype=float), [10, 50, 95, 99, 99.9])
    speed = out["speed_px_per_frame"]
    out["speed_band"] = "unknown"
    out.loc[speed <= q10, "speed_band"] = "still_or_slow"
    out.loc[(speed > q10) & (speed <= q50), "speed_band"] = "basal"
    out.loc[(speed > q50) & (speed <= q95), "speed_band"] = "mid"
    out.loc[(speed > q95) & (speed <= q99), "speed_band"] = "fast"
    out.loc[(speed > q99) & (speed <= q999), "speed_band"] = "very_fast"
    out.loc[speed > q999, "speed_band"] = "extreme"
    out["speed_upper_q95"] = float(q95)
    out["speed_upper_q99"] = float(q99)
    out["speed_upper_q999"] = float(q999)
    return out


def summarize_motion_timeseries(ts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [
        "speed_px_per_frame",
        "angular_velocity_rad_per_frame",
        "heading_change_rad",
        "tangent_accel_px_per_frame2",
        "lateral_accel_px_per_frame2",
        "nearest_neighbor_distance_px",
    ]
    for metric in metrics:
        values = pd.to_numeric(ts[metric], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            continue
        row = {"metric": metric, "n": int(len(values))}
        for q in [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]:
            row[f"q{str(q).replace('.', '_')}"] = float(np.nanpercentile(values, q))
        row["max"] = float(np.nanmax(values))
        rows.append(row)
    return pd.DataFrame(rows)


def save_plots(ts: pd.DataFrame, out_dir: Path, *, label: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not available; skipping plots")
        return

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    colors = plt.cm.tab10.colors

    fig, axes = plt.subplots(5, 1, figsize=(16, 12), sharex=True)
    series = [
        ("speed_px_per_frame", "speed px/frame"),
        ("angular_velocity_rad_per_frame", "angular velocity rad/frame"),
        ("tangent_accel_px_per_frame2", "tangent accel px/frame^2"),
        ("lateral_accel_px_per_frame2", "lateral accel px/frame^2"),
        ("nearest_neighbor_distance_px", "nearest neighbor px"),
    ]
    for ax, (col, ylabel) in zip(axes, series):
        for i, (tid, sub) in enumerate(ts.groupby("track_id", sort=True)):
            ax.plot(sub["frame"], sub[col], lw=0.6, alpha=0.85, color=colors[i % len(colors)], label=f"track {tid}")
        if col == "nearest_neighbor_distance_px":
            close = ts.loc[ts["close_contact"].fillna(False), "frame"]
            if not close.empty:
                ax.scatter(close, np.full(len(close), 0.0), s=2, color="red", alpha=0.25, label="close")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("frame")
    fig.suptitle(label)
    fig.tight_layout()
    fig.savefig(plot_dir / "motion_timeseries.png", dpi=140)
    plt.close(fig)

    hist_cols = [
        "speed_px_per_frame",
        "angular_velocity_rad_per_frame",
        "heading_change_rad",
        "tangent_accel_px_per_frame2",
        "lateral_accel_px_per_frame2",
    ]
    fig, axes = plt.subplots(1, len(hist_cols), figsize=(20, 4))
    for ax, col in zip(axes, hist_cols):
        values = pd.to_numeric(ts[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        ax.hist(values, bins=80, color="#4c78a8", alpha=0.85)
        ax.set_title(col)
        ax.set_yscale("log")
        ax.grid(alpha=0.2)
    fig.suptitle(f"{label} distributions")
    fig.tight_layout()
    fig.savefig(plot_dir / "motion_distributions.png", dpi=140)
    plt.close(fig)


def write_outputs(ts: pd.DataFrame, out_dir: Path, *, label: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts.to_csv(out_dir / "motion_timeseries.csv", index=False)
    summary = summarize_motion_timeseries(ts)
    summary.to_csv(out_dir / "motion_summary.csv", index=False)
    thresholds = {
        "label": label,
        "speed_upper_q95": float(ts["speed_upper_q95"].dropna().iloc[0]) if ts["speed_upper_q95"].notna().any() else None,
        "speed_upper_q99": float(ts["speed_upper_q99"].dropna().iloc[0]) if ts["speed_upper_q99"].notna().any() else None,
        "speed_upper_q999": float(ts["speed_upper_q999"].dropna().iloc[0]) if ts["speed_upper_q999"].notna().any() else None,
        "speed_band_counts": ts["speed_band"].value_counts(dropna=False).to_dict(),
        "close_contact_rows": int(ts["close_contact"].fillna(False).sum()),
    }
    (out_dir / "weak_thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    save_plots(ts, out_dir, label=label)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compute motion time-series diagnostics for 2D track CSVs.")
    p.add_argument("--tracks", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--label", default=None)
    p.add_argument("--close-contact-distance", type=float, default=35.0)
    p.add_argument("--windows", type=int, nargs="+", default=[15, 45, 90])
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    label = args.label or args.tracks.stem
    tracks = pd.read_csv(args.tracks)
    ts = compute_motion_timeseries(
        tracks,
        close_contact_distance_px=args.close_contact_distance,
        windows=tuple(args.windows),
    )
    write_outputs(ts, args.out_dir, label=label)
    print(f"[done] wrote motion diagnostics to {args.out_dir}")


if __name__ == "__main__":
    main()
