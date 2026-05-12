from __future__ import annotations

"""
Evaluate 2D ID-tracking consistency across SLEAP keypoint schemas and views.

Reads the tracks.csv files produced by `track_simple_nn.py` for each
(variant, view) cell plus the underlying .slp files, then computes proxy
metrics for ID-tracking quality without ground truth. Outputs:

- id_eval_metrics.parquet : long-form (variant, view, metric, value)
- id_eval_comparison.csv  : wide table (metric x variant_view)
- id_eval_report.md       : human-readable summary with winners highlighted

Metrics (per variant x view):
- n_track_ids_total       : distinct track_ids issued (3 = perfect, higher = swaps)
- fragmentation_index     : n_track_ids_total / num_individuals
- expected_id_coverage    : fraction of frames with exactly num_individuals IDs
- detection_rate          : fraction of frames with >=1 detection
- mean_track_lifetime     : average frames per track
- gap_rate                : fraction of (track, frame) pairs that are gaps inside lifetime

Mouth-speed spike metrics (cross-schema-comparable: uses the mouth keypoint
position, not centroid; reflection<->fish jumps and ID-swap jumps both show
up as physically-implausible velocity spikes on the mouth):
- mouth_speed_p99 / p999  : 99th/99.9th percentile of consecutive-frame mouth jump (px)
- n_spikes_gt_<T>         : count of consecutive-frame jumps > T px (T in {15, 30, 60, 100})
- spike_rate_gt_<T>       : n_spikes / n_consecutive_transitions
- max_spike_free_run      : longest contiguous spike-free segment (T=30 px) across all tracks
- spike_free_track_count  : number of tracks with zero spikes (T=30 px)

Body-axis metrics (mouth+tail only):
- inter_track_collision_rate : fraction of frames where any two tracks are <5 px apart
- body_axis_continuity    : circular std of per-frame d(orientation)
- mouth_tail_flip_rate    : fraction of frames with >90 deg orientation flip
"""

import argparse
import functools
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))


SPIKE_THRESHOLDS_PX = (15, 30, 60, 100)
DEFAULT_SPIKE_T = 30  # px; jumps above this on consecutive frames are flagged
# Per-frame pixel distance under which two tracks are considered overlapping
# (used as a sanity-check; real two-fish overlap should never go this small).
COLLISION_THRESHOLD_PX = 5.0


@dataclass(frozen=True)
class Cell:
    variant: str
    view: str
    tracks_csv: Path
    slp_path: Path
    has_tail: bool


def _wrap_pi(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def _circular_std(deltas: np.ndarray) -> float:
    if deltas.size == 0:
        return float("nan")
    s = np.sin(deltas).mean()
    c = np.cos(deltas).mean()
    r = np.sqrt(s * s + c * c)
    if r <= 0:
        return float("nan")
    return float(np.sqrt(-2.0 * np.log(min(r, 1.0))))


def _track_metrics(tracks: pd.DataFrame, num_individuals: int) -> dict:
    """Schema-agnostic metrics from a tracks.csv (frame, track_id, x, y)."""
    if tracks.empty:
        return {
            "n_track_ids_total": 0,
            "fragmentation_index": float("nan"),
            "expected_id_coverage": 0.0,
            "detection_rate": 0.0,
            "mean_track_lifetime": float("nan"),
            "gap_rate": float("nan"),
            "velocity_jump_p99": float("nan"),
            "velocity_jump_p999": float("nan"),
            "inter_track_collision_rate": float("nan"),
        }

    frames = tracks["frame"].to_numpy()
    fmin, fmax = int(frames.min()), int(frames.max())

    ids = tracks["track_id"].unique()
    n_ids = int(len(ids))

    counts_per_frame = tracks.groupby("frame")["track_id"].nunique()
    full_idx = pd.RangeIndex(fmin, fmax + 1)
    counts_per_frame = counts_per_frame.reindex(full_idx, fill_value=0)
    expected_cov = float((counts_per_frame == num_individuals).mean())
    det_rate = float((counts_per_frame > 0).mean())

    lifetimes = []
    gap_pairs = 0
    obs_pairs = 0
    for tid, sub in tracks.groupby("track_id"):
        f = sub["frame"].to_numpy()
        if f.size == 0:
            continue
        life = int(f.max() - f.min() + 1)
        lifetimes.append(life)
        obs_pairs += life
        gap_pairs += life - len(f)
    mean_lifetime = float(np.mean(lifetimes)) if lifetimes else float("nan")
    gap_rate = float(gap_pairs / obs_pairs) if obs_pairs > 0 else float("nan")

    jumps_all = []
    for tid, sub in tracks.groupby("track_id"):
        sub = sub.sort_values("frame")
        f = sub["frame"].to_numpy()
        x = sub["x"].to_numpy()
        y = sub["y"].to_numpy()
        if len(f) < 2:
            continue
        dx = np.diff(x)
        dy = np.diff(y)
        gap = np.diff(f)
        mask = gap == 1
        if mask.any():
            jumps_all.append(np.sqrt(dx[mask] ** 2 + dy[mask] ** 2))
    if jumps_all:
        all_j = np.concatenate(jumps_all)
        p99 = float(np.percentile(all_j, 99))
        p999 = float(np.percentile(all_j, 99.9))
    else:
        p99 = p999 = float("nan")

    pivot = tracks.pivot_table(index="frame", columns="track_id", values=["x", "y"])
    collisions = 0
    n_frames = 0
    for fi, row in pivot.iterrows():
        xs = row["x"].dropna().to_numpy()
        ys = row["y"].dropna().to_numpy()
        n_frames += 1
        if len(xs) < 2:
            continue
        pts = np.stack([xs, ys], axis=1)
        diff = pts[:, None, :] - pts[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=-1))
        np.fill_diagonal(dist, np.inf)
        if (dist < COLLISION_THRESHOLD_PX).any():
            collisions += 1
    coll_rate = float(collisions / n_frames) if n_frames > 0 else float("nan")

    return {
        "n_track_ids_total": n_ids,
        "fragmentation_index": n_ids / float(num_individuals),
        "expected_id_coverage": expected_cov,
        "detection_rate": det_rate,
        "mean_track_lifetime": mean_lifetime,
        "gap_rate": gap_rate,
        "velocity_jump_p99": p99,
        "velocity_jump_p999": p999,
        "inter_track_collision_rate": coll_rate,
    }


@functools.lru_cache(maxsize=16)
def _load_frame_index(slp_path_str: str):
    """Cached: return (frame_to_instances, mouth_idx, tail_idx) for a .slp.

    The per-frame instance dict is the expensive part; caching it avoids
    rebuilding it for every (window x cell) pair when the batch driver runs
    many windows against the same .slp.
    """
    import sleap_io as sio

    labels = sio.load_slp(slp_path_str)
    node_names = [n.name for n in labels.skeletons[0].nodes]
    try:
        mi = node_names.index("mouth")
        ti = node_names.index("tail")
    except ValueError:
        raise RuntimeError(f"No mouth/tail nodes in {slp_path_str}: {node_names}")

    frame_to_instances: dict[int, list[np.ndarray]] = {}
    for lf in labels.labeled_frames:
        fi = int(lf.frame_idx)
        frame_to_instances[fi] = [inst.numpy() for inst in lf.instances]
    return frame_to_instances, mi, ti


def _attach_mouth_tail(tracks: pd.DataFrame, slp_path: Path) -> pd.DataFrame:
    """For mouth+tail .slp, attach (mouth_x, mouth_y, tail_x, tail_y) to each
    tracked row by nearest-centroid matching within the frame."""
    frame_to_instances, mi, ti = _load_frame_index(str(slp_path))

    mouth_x = np.full(len(tracks), np.nan)
    mouth_y = np.full(len(tracks), np.nan)
    tail_x = np.full(len(tracks), np.nan)
    tail_y = np.full(len(tracks), np.nan)

    frames = tracks["frame"].to_numpy()
    xs = tracks["x"].to_numpy()
    ys = tracks["y"].to_numpy()

    for i in range(len(tracks)):
        fi = int(frames[i])
        instances = frame_to_instances.get(fi, [])
        if not instances:
            continue
        best_d = np.inf
        best_pts = None
        for pts in instances:
            mask = np.isfinite(pts).all(axis=1)
            if not mask.any():
                continue
            centroid = pts[mask].mean(axis=0)
            d = (centroid[0] - xs[i]) ** 2 + (centroid[1] - ys[i]) ** 2
            if d < best_d:
                best_d = d
                best_pts = pts
        if best_pts is None:
            continue
        if np.isfinite(best_pts[mi]).all():
            mouth_x[i] = best_pts[mi, 0]
            mouth_y[i] = best_pts[mi, 1]
        if np.isfinite(best_pts[ti]).all():
            tail_x[i] = best_pts[ti, 0]
            tail_y[i] = best_pts[ti, 1]

    out = tracks.copy()
    out["mouth_x"] = mouth_x
    out["mouth_y"] = mouth_y
    out["tail_x"] = tail_x
    out["tail_y"] = tail_y
    return out


def _body_axis_metrics(tracks_with_axis: pd.DataFrame) -> dict:
    """body_axis_continuity + mouth_tail_flip_rate per track, averaged."""
    if "mouth_x" not in tracks_with_axis.columns:
        return {
            "body_axis_continuity": float("nan"),
            "mouth_tail_flip_rate": float("nan"),
        }

    cont = []
    flips = 0
    pairs = 0
    for tid, sub in tracks_with_axis.groupby("track_id"):
        sub = sub.sort_values("frame")
        f = sub["frame"].to_numpy()
        mx = sub["mouth_x"].to_numpy()
        my = sub["mouth_y"].to_numpy()
        tx = sub["tail_x"].to_numpy()
        ty = sub["tail_y"].to_numpy()

        valid = np.isfinite(mx) & np.isfinite(my) & np.isfinite(tx) & np.isfinite(ty)
        if valid.sum() < 3:
            continue
        f = f[valid]
        ang = np.arctan2(ty[valid] - my[valid], tx[valid] - mx[valid])

        gap = np.diff(f)
        consec = gap == 1
        if not consec.any():
            continue
        d_ang = _wrap_pi(np.diff(ang))
        d_ang = d_ang[consec]
        cont.append(_circular_std(d_ang))
        pairs += d_ang.size
        flips += int((np.abs(d_ang) > (np.pi / 2)).sum())

    return {
        "body_axis_continuity": float(np.nanmean(cont)) if cont else float("nan"),
        "mouth_tail_flip_rate": float(flips / pairs) if pairs > 0 else float("nan"),
    }


def _mouth_speed_series(tracks: pd.DataFrame) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Return {track_id: (frame_array, speed_array)} on the mouth keypoint.

    For mouth+tail tracks, mouth_x/mouth_y come from _attach_mouth_tail.
    For mouth-only tracks, the x,y in tracks.csv IS the mouth (single node).
    Only consecutive-frame transitions are kept (gap-aware).
    """
    use_mouth = "mouth_x" in tracks.columns and "mouth_y" in tracks.columns
    series: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for tid, sub in tracks.groupby("track_id"):
        sub = sub.sort_values("frame")
        f = sub["frame"].to_numpy()
        if use_mouth:
            x = sub["mouth_x"].to_numpy()
            y = sub["mouth_y"].to_numpy()
        else:
            x = sub["x"].to_numpy()
            y = sub["y"].to_numpy()
        valid = np.isfinite(x) & np.isfinite(y)
        f = f[valid]
        x = x[valid]
        y = y[valid]
        if len(f) < 2:
            series[int(tid)] = (np.empty(0, dtype=int), np.empty(0, dtype=float))
            continue
        gap = np.diff(f)
        consec = gap == 1
        if not consec.any():
            series[int(tid)] = (np.empty(0, dtype=int), np.empty(0, dtype=float))
            continue
        speeds = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)[consec]
        speed_frames = f[1:][consec]
        series[int(tid)] = (speed_frames, speeds)
    return series


def _spike_metrics(tracks: pd.DataFrame) -> dict:
    """Mouth-speed spike statistics (cross-schema-comparable)."""
    series = _mouth_speed_series(tracks)

    all_speeds = []
    for _, (_, sp) in series.items():
        if sp.size:
            all_speeds.append(sp)
    if not all_speeds:
        out = {
            "mouth_speed_p99": float("nan"),
            "mouth_speed_p999": float("nan"),
            "max_spike_free_run": 0,
            "spike_free_track_count": 0,
        }
        for T in SPIKE_THRESHOLDS_PX:
            out[f"n_spikes_gt_{T}"] = 0
            out[f"spike_rate_gt_{T}"] = float("nan")
        return out

    sp_concat = np.concatenate(all_speeds)
    p99 = float(np.percentile(sp_concat, 99))
    p999 = float(np.percentile(sp_concat, 99.9))

    out: dict[str, float] = {
        "mouth_speed_p99": p99,
        "mouth_speed_p999": p999,
    }
    total_transitions = int(sp_concat.size)
    for T in SPIKE_THRESHOLDS_PX:
        n = int((sp_concat > T).sum())
        out[f"n_spikes_gt_{T}"] = n
        out[f"spike_rate_gt_{T}"] = float(n / total_transitions)

    T = DEFAULT_SPIKE_T
    longest = 0
    n_clean_tracks = 0
    for tid, (_, sp) in series.items():
        if sp.size == 0:
            continue
        is_clean = sp <= T
        if is_clean.all():
            n_clean_tracks += 1
        run = 0
        for v in is_clean:
            run = run + 1 if v else 0
            if run > longest:
                longest = run
    out["max_spike_free_run"] = int(longest)
    out["spike_free_track_count"] = int(n_clean_tracks)
    return out


def _save_speed_plot(tracks: pd.DataFrame, title: str, out_path: Path) -> None:
    """Render per-track mouth-speed time series with the default spike threshold."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"[warn] matplotlib not available; skipping plot for {title}")
        return
    series = _mouth_speed_series(tracks)
    fig, ax = plt.subplots(figsize=(14, 4))
    colors = plt.cm.tab10.colors
    for i, (tid, (f, sp)) in enumerate(sorted(series.items())):
        if sp.size == 0:
            continue
        ax.plot(f, sp, lw=0.5, color=colors[i % len(colors)],
                label=f"track_id={tid} (n={sp.size})")
    ax.axhline(DEFAULT_SPIKE_T, color="red", lw=0.8, ls="--",
               label=f"spike threshold = {DEFAULT_SPIKE_T} px")
    ax.set_xlabel("frame")
    ax.set_ylabel("mouth speed (px / frame)")
    ax.set_title(title)
    ax.set_ylim(0, max(100, ax.get_ylim()[1]))
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate(cells: list[Cell], num_individuals: int = 3,
             plot_dir: Path | None = None) -> pd.DataFrame:
    rows = []
    for c in cells:
        print(f"[eval] {c.variant} / {c.view}")
        tracks = pd.read_csv(c.tracks_csv)
        base = _track_metrics(tracks, num_individuals)
        if c.has_tail and c.slp_path.exists():
            tracks_with_axis = _attach_mouth_tail(tracks, c.slp_path)
            axis = _body_axis_metrics(tracks_with_axis)
        else:
            tracks_with_axis = tracks
            axis = {"body_axis_continuity": float("nan"),
                    "mouth_tail_flip_rate": float("nan")}
        spike = _spike_metrics(tracks_with_axis)
        if plot_dir is not None:
            _save_speed_plot(tracks_with_axis,
                             title=f"{c.variant} / {c.view}",
                             out_path=plot_dir / f"{c.variant}__{c.view}__speed.png")
        m = {**base, **axis, **spike}
        for k, v in m.items():
            rows.append({"variant": c.variant, "view": c.view,
                         "metric": k, "value": v})
    return pd.DataFrame(rows)


def _winner_per_metric(comparison: pd.DataFrame) -> dict[str, str]:
    higher = {"expected_id_coverage", "detection_rate", "mean_track_lifetime",
              "max_spike_free_run", "spike_free_track_count"}
    lower = {"n_track_ids_total", "fragmentation_index", "gap_rate",
             "velocity_jump_p99", "velocity_jump_p999",
             "inter_track_collision_rate", "body_axis_continuity",
             "mouth_tail_flip_rate",
             "mouth_speed_p99", "mouth_speed_p999"}
    for T in SPIKE_THRESHOLDS_PX:
        lower.add(f"n_spikes_gt_{T}")
        lower.add(f"spike_rate_gt_{T}")
    winners: dict[str, str] = {}
    for metric, row in comparison.iterrows():
        vals = row.dropna()
        if vals.empty:
            continue
        if metric in higher:
            winners[metric] = str(vals.idxmax())
        elif metric in lower:
            winners[metric] = str(vals.idxmin())
    return winners


def write_outputs(metrics_df: pd.DataFrame, out_dir: Path,
                  num_individuals: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        metrics_df.to_parquet(out_dir / "id_eval_metrics.parquet", index=False)
    except Exception as e:
        print(f"[warn] parquet write failed ({e}); falling back to CSV")
        metrics_df.to_csv(out_dir / "id_eval_metrics.csv", index=False)

    metrics_df = metrics_df.copy()
    metrics_df["col"] = metrics_df["variant"] + "__" + metrics_df["view"]
    wide = metrics_df.pivot_table(index="metric", columns="col", values="value")

    metric_order = [
        # Primary real-fish-tracking quality: mouth-speed spikes
        "mouth_speed_p99",
        "mouth_speed_p999",
    ] + [f"n_spikes_gt_{T}" for T in SPIKE_THRESHOLDS_PX] + [
        f"spike_rate_gt_{T}" for T in SPIKE_THRESHOLDS_PX
    ] + [
        "max_spike_free_run",
        "spike_free_track_count",
        # Secondary: ID-consistency metrics
        "n_track_ids_total",
        "fragmentation_index",
        "expected_id_coverage",
        "detection_rate",
        "mean_track_lifetime",
        "gap_rate",
        "velocity_jump_p99",
        "velocity_jump_p999",
        "inter_track_collision_rate",
        "body_axis_continuity",
        "mouth_tail_flip_rate",
    ]
    wide = wide.reindex([m for m in metric_order if m in wide.index])

    views_order = ["top_cam", "side1_cam", "side2_cam"]
    variants_order = ["mouth_only_n3", "mouth_tail_n3",
                      "mouth_tail_n6", "mouth_tail_n6_roi"]
    col_order = [f"{v}__{view}" for view in views_order for v in variants_order
                 if f"{v}__{view}" in wide.columns]
    wide = wide[col_order]
    wide.to_csv(out_dir / "id_eval_comparison.csv")

    winners = _winner_per_metric(wide)
    fmt = {
        "n_track_ids_total": "{:.0f}",
        "fragmentation_index": "{:.2f}",
        "expected_id_coverage": "{:.3f}",
        "detection_rate": "{:.3f}",
        "mean_track_lifetime": "{:.0f}",
        "gap_rate": "{:.4f}",
        "velocity_jump_p99": "{:.1f}",
        "velocity_jump_p999": "{:.1f}",
        "inter_track_collision_rate": "{:.4f}",
        "body_axis_continuity": "{:.3f}",
        "mouth_tail_flip_rate": "{:.4f}",
        "mouth_speed_p99": "{:.1f}",
        "mouth_speed_p999": "{:.1f}",
        "max_spike_free_run": "{:.0f}",
        "spike_free_track_count": "{:.0f}",
    }
    for T in SPIKE_THRESHOLDS_PX:
        fmt[f"n_spikes_gt_{T}"] = "{:.0f}"
        fmt[f"spike_rate_gt_{T}"] = "{:.4f}"

    lines = []
    lines.append("# 2D ID-tracking consistency evaluation")
    lines.append("")
    lines.append(f"Num individuals expected: **{num_individuals}**.")
    lines.append("")
    lines.append("## Comparison table")
    lines.append("")
    lines.append("| metric | " + " | ".join(wide.columns) + " | winner |")
    lines.append("|---|" + "|".join("---" for _ in wide.columns) + "|---|")
    for metric, row in wide.iterrows():
        cells_md = []
        win_col = winners.get(metric, "")
        for col, val in row.items():
            if pd.isna(val):
                cells_md.append("—")
            else:
                s = fmt.get(metric, "{:.3f}").format(val)
                if col == win_col:
                    s = f"**{s}**"
                cells_md.append(s)
        lines.append(f"| {metric} | " + " | ".join(cells_md) + f" | {win_col} |")
    lines.append("")

    lines.append("## Quick read")
    lines.append("")
    win_count = pd.Series(winners).value_counts()
    lines.append("Wins per cell (across all metrics):")
    lines.append("")
    for col, n in win_count.items():
        lines.append(f"- `{col}` : {int(n)}")
    lines.append("")

    sch_score = {"mouth_only_n3": 0, "mouth_tail_n3": 0,
                 "mouth_tail_n6": 0, "mouth_tail_n6_roi": 0}
    for col, n in win_count.items():
        for s in sch_score:
            if col.startswith(s + "__"):
                sch_score[s] += int(n)
    lines.append("Wins per **schema** (summed across views):")
    lines.append("")
    for s, n in sorted(sch_score.items(), key=lambda kv: -kv[1]):
        lines.append(f"- `{s}` : {n}")
    lines.append("")

    view_score = {"top_cam": 0, "side1_cam": 0, "side2_cam": 0}
    for col, n in win_count.items():
        for v in view_score:
            if col.endswith("__" + v):
                view_score[v] += int(n)
    lines.append("Wins per **view** (summed across schemas):")
    lines.append("")
    for v, n in sorted(view_score.items(), key=lambda kv: -kv[1]):
        lines.append(f"- `{v}` : {n}")
    lines.append("")

    (out_dir / "id_eval_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[done] outputs in {out_dir}")
    for p in sorted(out_dir.glob("id_eval*")):
        print(f"  {p.name}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Score 2D ID-tracking consistency across schemas and views."
    )
    p.add_argument("--run-dir", required=True,
                   help="Directory containing <variant>__<view>.tracks.csv files.")
    p.add_argument("--slp-root", required=True,
                   help="Root sleap_output folder, e.g. "
                        "Z:/StephensU/medaka/sleap_output/<experiment>.")
    p.add_argument("--num-individuals", type=int, default=3)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_dir = Path(args.run_dir)
    slp_root = Path(args.slp_root)

    layout = [
        ("mouth_only_n3",     slp_root,                   "", False),
        ("mouth_tail_n3",     slp_root / "round2",        "", True),
        ("mouth_tail_n6",     slp_root / "round2_trt_n6", "", True),
        ("mouth_tail_n6_roi", slp_root / "round2_trt_n6", ".tank_filtered", True),
    ]
    views = ["top_cam", "side1_cam", "side2_cam"]
    cells: list[Cell] = []
    for variant, root, suffix, has_tail in layout:
        for view in views:
            tracks_csv = run_dir / f"{variant}__{view}.tracks.csv"
            slp_path = root / f"{view}{suffix}.slp"
            if not tracks_csv.exists():
                print(f"[skip] {tracks_csv} missing")
                continue
            cells.append(Cell(variant=variant, view=view,
                              tracks_csv=tracks_csv, slp_path=slp_path,
                              has_tail=has_tail))

    plot_dir = run_dir / "speed_plots"
    metrics_df = evaluate(cells, num_individuals=args.num_individuals,
                          plot_dir=plot_dir)
    write_outputs(metrics_df, run_dir, num_individuals=args.num_individuals)


if __name__ == "__main__":
    main()
