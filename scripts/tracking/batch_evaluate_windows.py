from __future__ import annotations

"""
Run the full 2D ID-tracking evaluation over multiple non-overlapping
frame windows of one experiment, then aggregate.

For each window:
  1. NN-track all (variant x view) cells using preloaded .slp files
     (`len(VARIANT_LAYOUT) x len(VIEWS)` cells per window, with .slp files
     shared across variants pointing at the same path so each is loaded once).
  2. Score with `evaluate_id_consistency.evaluate()` -> metrics + speed plots.
  3. Render top_cam trajectory PNGs (mouth+tail n=3 vs n=6 vs mouth_only).

After all windows: write `aggregate/metrics_long_all_windows.parquet`,
`aggregate/metrics_summary_across_windows.csv`, and per-metric box-plots.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.tracking.evaluate_id_consistency import (  # noqa: E402
    Cell, evaluate, write_outputs,
)


VARIANT_LAYOUT = [
    # (key, subdir, slp_suffix, has_tail, tracker_extras)
    ("mouth_only_n3",        "",              "",               False, {}),
    ("mouth_tail_n3",        "round2",        "",               True,  {}),
    ("mouth_tail_n6",        "round2_trt_n6", "",               True,  {}),
    ("mouth_tail_n6_roi",    "round2_trt_n6", ".tank_filtered", True,  {}),
    # Step-2 candidate: mouth-anchored matching + score-filtered top-3 on n=6 source.
    ("mouth_anchored_top3",  "round2_trt_n6", "",               True,
        {"anchor_node": "mouth", "top_k_by_score": 3}),
    # Step-2 isolation A: mouth-anchored only (no score filter) — isolates mouth-vs-centroid effect.
    ("mouth_anchored_n6",    "round2_trt_n6", "",               True,
        {"anchor_node": "mouth"}),
    # Step-2 isolation B: score-filter only (centroid) — isolates top-K-vs-all effect.
    ("top3_score_n6",        "round2_trt_n6", "",               True,
        {"top_k_by_score": 3}),
]
VIEWS = ["top_cam", "side1_cam", "side2_cam"]


def _instance_position(inst, *, node_names=None, anchor_node=None):
    pts = inst.numpy()
    if anchor_node is not None and node_names is not None:
        try:
            idx = node_names.index(anchor_node)
            pt = pts[idx]
            if np.isfinite(pt).all():
                return float(pt[0]), float(pt[1])
        except ValueError:
            pass
    mask = np.isfinite(pts).all(axis=1)
    if not mask.any():
        return None
    centroid = pts[mask].mean(axis=0)
    return float(centroid[0]), float(centroid[1])


def track_in_memory(labels, *, start_frame, end_frame,
                    max_distance=100.0, max_tracks=3, max_gap=500,
                    anchor_node=None, top_k_by_score=None):
    """Same Hungarian-NN algorithm as track_simple_nn.track(), but on a
    preloaded Labels object and a frame range. Supports the same anchor_node
    and top_k_by_score knobs as the CLI tracker."""
    node_names = [n.name for n in labels.skeletons[0].nodes]
    node_name = anchor_node if (anchor_node in node_names) else node_names[0]
    frames = [f for f in labels.labeled_frames
              if start_frame <= int(f.frame_idx) <= end_frame]
    frames.sort(key=lambda f: int(f.frame_idx))

    active: dict[int, dict] = {}
    next_id = 0
    rows: list[dict] = []

    for f in frames:
        fi = int(f.frame_idx)
        stale = [tid for tid, tr in active.items()
                 if (fi - tr["last_frame"]) > max_gap]
        for tid in stale:
            del active[tid]

        instances = list(f.instances)
        if top_k_by_score is not None and len(instances) > top_k_by_score:
            instances = sorted(
                instances,
                key=lambda inst: (getattr(inst, "score", None) or 0.0),
                reverse=True,
            )[:top_k_by_score]

        detections = []
        for inst in instances:
            pos = _instance_position(
                inst, node_names=node_names, anchor_node=anchor_node,
            )
            if pos is None:
                continue
            score_raw = getattr(inst, "score", None)
            detections.append({
                "x": pos[0], "y": pos[1],
                "score": float(score_raw) if score_raw is not None else np.nan,
            })

        if not detections:
            continue

        if not active:
            for det in detections[:max_tracks]:
                tid = next_id
                next_id += 1
                active[tid] = {"last_frame": fi,
                               "last_pos": (det["x"], det["y"])}
                rows.append({"frame": fi, "track_id": tid,
                             "x": det["x"], "y": det["y"],
                             "score": det["score"], "node": node_name})
            continue

        track_ids = list(active.keys())
        track_pos = np.array([active[t]["last_pos"] for t in track_ids])
        det_pos = np.array([[d["x"], d["y"]] for d in detections])
        diff = track_pos[:, None, :] - det_pos[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=-1))
        cost = dist.copy()
        cost[dist > max_distance] = 1e9
        row_ind, col_ind = linear_sum_assignment(cost)

        matched_dets: set[int] = set()
        for ri, ci in zip(row_ind, col_ind):
            if dist[ri, ci] > max_distance:
                continue
            tid = track_ids[ri]
            det = detections[ci]
            active[tid]["last_frame"] = fi
            active[tid]["last_pos"] = (det["x"], det["y"])
            rows.append({"frame": fi, "track_id": tid,
                         "x": det["x"], "y": det["y"],
                         "score": det["score"], "node": node_name})
            matched_dets.add(ci)

        for ci, det in enumerate(detections):
            if ci in matched_dets:
                continue
            if len(active) >= max_tracks:
                continue
            tid = next_id
            next_id += 1
            active[tid] = {"last_frame": fi,
                           "last_pos": (det["x"], det["y"])}
            rows.append({"frame": fi, "track_id": tid,
                         "x": det["x"], "y": det["y"],
                         "score": det["score"], "node": node_name})

    return rows


def _grab_frame(video_path: str, frame_idx: int) -> np.ndarray | None:
    try:
        import cv2
    except ImportError:
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _render_trajectory_compare(win_dir: Path, video_path: str,
                               mid_frame: int, frame_range: tuple[int, int]):
    """Side-by-side trajectory render: n=3 vs n=6 vs mouth_only."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bg = _grab_frame(video_path, mid_frame)
    if bg is None:
        print(f"[warn] could not grab bg frame {mid_frame} from {video_path}")
        bg = np.full((1200, 1200, 3), 128, dtype=np.uint8)
    H, W = bg.shape[:2]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
              "#9467bd", "#8c564b", "#e377c2"]

    pairs = [
        ("mouth_tail_n3", "mouth+tail n=3"),
        ("mouth_tail_n6", "mouth+tail n=6"),
        ("mouth_only_n3", "mouth-only n=3"),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(33, 11))
    for ax, (variant, label) in zip(axs, pairs):
        csv = win_dir / f"{variant}__top_cam.tracks.csv"
        tracks = pd.read_csv(csv)
        ax.imshow(bg, alpha=0.55)
        for i, (tid, sub) in enumerate(sorted(tracks.groupby("track_id"))):
            sub = sub.sort_values("frame")
            c = colors[i % len(colors)]
            ax.plot(sub["x"], sub["y"], lw=0.7, color=c, alpha=0.85,
                    label=f"track {int(tid)} (n={len(sub)})")
            ax.scatter(sub["x"].iloc[0], sub["y"].iloc[0], s=70, c=c,
                       edgecolor="white", linewidth=1.5, marker="o", zorder=5)
            ax.scatter(sub["x"].iloc[-1], sub["y"].iloc[-1], s=100, c=c,
                       edgecolor="white", linewidth=1.8, marker="X", zorder=5)
        ax.set_title(label, fontsize=14)
        ax.set_xlim(0, W); ax.set_ylim(H, 0)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(loc="upper right", fontsize=10, framealpha=0.85)
    fig.suptitle(f"top_cam trajectories  frames {frame_range[0]:,}-{frame_range[1]:,}",
                 fontsize=14)
    fig.tight_layout()
    out_dir = win_dir / "trajectories"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "compare__top_cam.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def _aggregate(agg_dir: Path, all_metrics: list[pd.DataFrame]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agg = pd.concat(all_metrics, ignore_index=True)
    agg_dir.mkdir(parents=True, exist_ok=True)
    try:
        agg.to_parquet(agg_dir / "metrics_long_all_windows.parquet", index=False)
    except Exception:
        agg.to_csv(agg_dir / "metrics_long_all_windows.csv", index=False)

    summary = (agg.groupby(["variant", "view", "metric"])["value"]
               .agg(["mean", "std", "min", "max"])
               .reset_index())
    summary.to_csv(agg_dir / "metrics_summary_across_windows.csv", index=False)

    headline = ["n_spikes_gt_30", "n_spikes_gt_100",
                "mouth_speed_p999", "n_track_ids_total",
                "max_spike_free_run"]
    pivot = (agg[agg["metric"].isin(headline)]
             .assign(col=lambda d: d["variant"] + "__" + d["view"])
             .pivot_table(index=["window", "metric"], columns="col",
                          values="value")
             .reset_index())
    pivot.to_csv(agg_dir / "headline_metrics_per_window.csv", index=False)

    for metric in headline:
        sub = agg[agg["metric"] == metric].copy()
        if sub.empty:
            continue
        sub["cell"] = sub["variant"] + "__" + sub["view"]
        cell_order = []
        for view in VIEWS:
            for var, *_ in VARIANT_LAYOUT:
                cell_order.append(f"{var}__{view}")
        cell_order = [c for c in cell_order if c in sub["cell"].unique()]
        data_by_cell = [sub[sub["cell"] == c]["value"].dropna().to_numpy()
                        for c in cell_order]
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.boxplot(data_by_cell, labels=cell_order, showfliers=True,
                   widths=0.6)
        ax.set_title(f"{metric} across windows (n={agg['window'].nunique()})")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        for tick in ax.get_xticklabels():
            tick.set_ha("right")
        fig.tight_layout()
        fig.savefig(agg_dir / f"box__{metric}.png", dpi=130,
                    bbox_inches="tight")
        plt.close(fig)
    print(f"\nAggregate outputs in {agg_dir}")


def main(argv: list[str] | None = None):
    p = argparse.ArgumentParser()
    p.add_argument("--slp-root", required=True,
                   help="...sleap_output/<experiment>/")
    p.add_argument("--top-cam-video", required=True,
                   help="path to top_cam mp4 for trajectory backgrounds")
    p.add_argument("--out-root", required=True)
    p.add_argument("--windows", type=int, nargs="+", required=True,
                   help="window start frames; each spans start..start+window-size-1")
    p.add_argument("--window-size", type=int, default=10000,
                   help="frames per window (default 10000)")
    p.add_argument("--num-individuals", type=int, default=3)
    args = p.parse_args(argv)

    slp_root = Path(args.slp_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    import sleap_io as sio
    print(f"=== Pre-loading {len(VARIANT_LAYOUT) * len(VIEWS)} .slp files...")
    labels_cache: dict[tuple[str, str], object] = {}
    slp_paths: dict[tuple[str, str], Path] = {}
    has_tail_map: dict[tuple[str, str], bool] = {}
    extras_map: dict[tuple[str, str], dict] = {}
    # Cache loaded labels by absolute slp path so two variants pointing at the
    # same .slp (e.g. mouth_tail_n6 + mouth_anchored_top3) reuse one load.
    by_path: dict[Path, object] = {}
    for variant, subdir, suffix, has_tail, extras in VARIANT_LAYOUT:
        for view in VIEWS:
            sub = slp_root / subdir if subdir else slp_root
            slp_path = sub / f"{view}{suffix}.slp"
            if slp_path not in by_path:
                print(f"  load {variant}/{view}: {slp_path}")
                by_path[slp_path] = sio.load_slp(str(slp_path))
            else:
                print(f"  reuse {variant}/{view}: {slp_path}")
            labels_cache[(variant, view)] = by_path[slp_path]
            slp_paths[(variant, view)] = slp_path
            has_tail_map[(variant, view)] = has_tail
            extras_map[(variant, view)] = dict(extras)
    print("=== Pre-load complete.")

    all_metrics: list[pd.DataFrame] = []

    for start in args.windows:
        end = start + args.window_size - 1
        win_label = f"f{start:06d}-{end:06d}"
        win_dir = out_root / win_label
        win_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Window {win_label}")

        for (variant, view), labels in labels_cache.items():
            extras = extras_map[(variant, view)]
            rows = track_in_memory(
                labels, start_frame=start, end_frame=end, **extras,
            )
            df = pd.DataFrame(rows,
                              columns=["frame", "track_id", "x", "y",
                                       "score", "node"])
            df.to_csv(win_dir / f"{variant}__{view}.tracks.csv", index=False)
            n_tracks = int(df["track_id"].nunique()) if not df.empty else 0
            print(f"  {variant}/{view}: rows={len(df)} tracks={n_tracks}")

        cells = [
            Cell(variant=variant, view=view,
                 tracks_csv=win_dir / f"{variant}__{view}.tracks.csv",
                 slp_path=slp_paths[(variant, view)],
                 has_tail=has_tail_map[(variant, view)])
            for variant in [v for v, *_ in VARIANT_LAYOUT]
            for view in VIEWS
        ]
        metrics_df = evaluate(cells, num_individuals=args.num_individuals,
                              plot_dir=win_dir / "speed_plots")
        write_outputs(metrics_df, win_dir,
                      num_individuals=args.num_individuals)

        mid_frame = (start + end) // 2
        _render_trajectory_compare(win_dir, args.top_cam_video,
                                   mid_frame, (start, end))

        metrics_df = metrics_df.copy()
        metrics_df["window"] = win_label
        metrics_df["window_start"] = start
        all_metrics.append(metrics_df)

    _aggregate(out_root / "aggregate", all_metrics)


if __name__ == "__main__":
    main()
