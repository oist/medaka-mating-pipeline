from __future__ import annotations

"""Orchestrate the tracklet -> 3-fish identity benchmark.

Reads a TOML config, selects pilot windows, loads every available method into the
common AssignmentTable, computes the metric panel + inter-method agreement +
(when labels exist) GT scoring, and writes an inspectable results directory plus a
prioritised annotation worklist. Resumable and dry-runnable. See README.md.

    python -m scripts.experiments.tracklet_id_benchmark.run_benchmark \
        --config config/experiments/tracklet_id_benchmark.example.toml [--dry-run]
"""

import argparse
import json
import logging
import sys
import tomllib
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from scripts.common.local_paths import load_local_paths
from scripts.experiments.tracklet_id_benchmark import figures, methods, metrics
from scripts.experiments.tracklet_id_benchmark.windows import select_windows

LOG = logging.getLogger("tracklet_id_benchmark")
ALL_METHODS = ["geometry_3d", "tracklet_bind", "motion_resolver", "hybrid", "appearance_idtracker"]


def _resolve(path_str, repo_root: Path) -> str | None:
    if not path_str:
        return None
    p = Path(path_str)
    return str(p if p.is_absolute() else (repo_root / p))


def _json_safe(o):
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the tracklet->3-fish identity benchmark.")
    p.add_argument("--config", required=True)
    p.add_argument("--out-dir", default=None, help="override results root from config")
    p.add_argument("--run-tag", default=None)
    p.add_argument("--dry-run", action="store_true", help="select windows + list trials, compute nothing")
    p.add_argument("--max-windows", type=int, default=None)
    p.add_argument("--kinds", nargs="*", default=None, help="only these window kinds")
    p.add_argument("--methods", nargs="*", default=None, help="subset of methods")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--force", action="store_true", help="recompute existing trials")
    p.add_argument("--paths-config", default=None)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[3]
    cfg = tomllib.loads(Path(args.config).read_text(encoding="utf-8"))
    bm = cfg.get("benchmark", {})
    local = load_local_paths(args.paths_config)

    # --- resolve inputs ---
    csv_3d = _resolve(bm.get("csv_3d"), repo_root)
    bind_dir = _resolve(bm.get("bind_dir"), repo_root)
    resolved_csv = _resolve(bm.get("resolved_events_csv"), repo_root)
    appearance_csv = _resolve(bm.get("appearance_traj_csv"), repo_root)
    out_root = Path(args.out_dir or _resolve(bm.get("out_dir"), repo_root)
                    or (local.runs_dir if local else "runs"))
    run_tag = args.run_tag or bm.get("run_tag", "pilot")
    out_dir = out_root / "tracklet_id_benchmark" / bm.get("experiment", "exp") / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(out_dir / "run.log", encoding="utf-8"),
                                  logging.StreamHandler()])
    LOG.info("config=%s  out_dir=%s", args.config, out_dir)
    for label, pth in [("csv_3d", csv_3d), ("resolved_events_csv", resolved_csv)]:
        if not pth or not Path(pth).exists():
            raise SystemExit(f"required input '{label}' missing: {pth}")

    # --- select windows ---
    resolved = pd.read_csv(resolved_csv)
    d3 = methods.load_3d(csv_3d)
    total_frames = int(d3["frame"].max()) + 1
    wsel = bm.get("windows", {})
    windows = select_windows(resolved, total_frames=total_frames,
                             fps=bm.get("fps", 166.0),
                             pad_frames=wsel.get("pad_frames", 500),
                             n_suspect=wsel.get("n_suspect", 4),
                             early_fp_end=wsel.get("early_fp_end", 8000))
    if args.kinds:
        windows = [w for w in windows if w["kind"] in args.kinds]
    if args.max_windows:
        windows = windows[: args.max_windows]
    pd.DataFrame(windows).to_csv(out_dir / "windows.csv", index=False)
    LOG.info("selected %d windows: %s", len(windows), [w["name"] for w in windows])

    method_list = args.methods or bm.get("methods", ALL_METHODS)
    events = resolved.to_dict("records")

    resolved_cfg = {"config": cfg, "csv_3d": csv_3d, "bind_dir": bind_dir,
                    "resolved_events_csv": resolved_csv, "appearance_traj_csv": appearance_csv,
                    "methods": method_list, "total_frames": total_frames,
                    "n_windows": len(windows)}
    (out_dir / "config.resolved.json").write_text(json.dumps(_json_safe(resolved_cfg), indent=2),
                                                  encoding="utf-8")

    if args.dry_run:
        LOG.info("DRY RUN — %d windows x %d methods = %d trials",
                 len(windows), len(method_list), len(windows) * len(method_list))
        print(pd.DataFrame(windows)[["name", "kind", "start", "end", "n_frames"]].to_string(index=False))
        print("\nmethods:", method_list)
        return

    spike_mm = bm.get("spike_mm", 8.0)
    n_fish = bm.get("n_fish", 3)
    gt_csv = _resolve(bm.get("gt_csv"), repo_root)
    gt = pd.read_csv(gt_csv) if gt_csv and Path(gt_csv).exists() else None

    summary_rows = []
    (out_dir / "trials").mkdir(exist_ok=True)
    (out_dir / "agreement.csv").unlink(missing_ok=True)  # rebuilt fresh each run
    for w in windows:
        tables = _build_tables(d3, bind_dir, resolved, appearance_csv, method_list, w)
        for m, df in tables.items():
            if df is None:
                LOG.warning("[%s] method %s unavailable", w["name"], m)
                continue
            trial_path = out_dir / "trials" / f"{w['name']}__{m}.json"
            if trial_path.exists() and not args.force:
                rec = json.loads(trial_path.read_text(encoding="utf-8"))
            else:
                panel = metrics.window_panel(df, w, events, n_fish=n_fish, spike_mm=spike_mm)
                swap_calls = methods.method_swap_calls(m, resolved, w["start"], w["end"])
                gt_score = metrics.gt_swap_accuracy(swap_calls, gt) if gt is not None else \
                    {"accuracy": float("nan"), "n_labeled": 0}
                rec = {"window": w["name"], "kind": w["kind"], "method": m,
                       "start": w["start"], "end": w["end"], **panel, "gt_swap": gt_score}
                trial_path.write_text(json.dumps(_json_safe(rec), indent=2), encoding="utf-8")
            summary_rows.append(_flatten(rec))

        # inter-method agreement (per window) + figures
        _agreement(tables, w, out_dir)
        if not args.no_figures:
            avail = {m: df for m, df in tables.items() if df is not None}
            if avail:
                figures.timeline(avail, w, events, str(out_dir / "figures" / f"{w['name']}_timeline.png"), n_fish)
            if "geometry_3d" in avail:
                figures.traj3d(avail["geometry_3d"], w, str(out_dir / "figures" / f"{w['name']}_traj3d.png"), n_fish)

    sdf = pd.DataFrame(summary_rows)
    sdf.to_csv(out_dir / "summary.csv", index=False)
    _write_summary_json(sdf, windows, out_dir, gt is not None)
    _write_worklist(windows, resolved, out_dir, gt)
    LOG.info("DONE -> %s", out_dir)
    if not sdf.empty:
        cols = [c for c in ["window", "method", "coverage", "unknown_frac",
                            "speed_spike_mean_per_k", "reconnection_rate"] if c in sdf.columns]
        print("\n=== summary (head) ===")
        print(sdf[cols].to_string(index=False))


def _build_tables(d3, bind_dir, resolved, appearance_csv, method_list, w):
    s, e = w["start"], w["end"]
    tables = {}
    for m in method_list:
        if m == "geometry_3d":
            tables[m] = methods.geometry_3d_table(d3, s, e)
        elif m == "tracklet_bind":
            if bind_dir and Path(bind_dir).exists():
                tables[m], _ = methods.tracklet_bind_table(d3, bind_dir, s, e)
            else:
                tables[m] = None
        elif m == "motion_resolver":
            tables[m] = methods.motion_resolver_table(d3, resolved, s, e)
        elif m == "hybrid":
            tables[m] = methods.hybrid_table(d3, bind_dir, resolved, s, e) if bind_dir else None
        elif m == "appearance_idtracker":
            t, reason = methods.appearance_idtracker_table(appearance_csv, s, e)
            tables[m] = t
            if t is None:
                LOG.info("[%s] appearance_idtracker: %s", w["name"], reason)
    return tables


def _agreement(tables, w, out_dir):
    avail = {m: df for m, df in tables.items() if df is not None}
    rows = []
    for a, b in combinations(avail, 2):
        ag = metrics.align_labels(avail[a], avail[b], w["start"], w["end"])
        rows.append({"window": w["name"], "method_a": a, "method_b": b, **ag})
    if rows:
        path = out_dir / "agreement.csv"
        df = pd.DataFrame(rows)
        header = not path.exists()
        df.to_csv(path, mode="a", header=header, index=False)


def _flatten(rec: dict) -> dict:
    out = {k: v for k, v in rec.items() if not isinstance(v, dict)}
    ss = rec.get("speed_spike", {})
    out["speed_spike_mean_per_k"] = ss.get("mean_per_k")
    rc = rec.get("reconnection", {})
    out["reconnection_rate"] = rc.get("reconnection_rate")
    out["n_distinct_ids"] = rec.get("group_n_distinct_ids")
    out["gt_swap_accuracy"] = rec.get("gt_swap", {}).get("accuracy")
    return out


def _write_summary_json(sdf, windows, out_dir, has_gt):
    agg = {}
    if not sdf.empty:
        for m, g in sdf.groupby("method"):
            agg[m] = {"mean_coverage": _safe_mean(g, "coverage"),
                      "mean_unknown_frac": _safe_mean(g, "unknown_frac"),
                      "mean_speed_spike_per_k": _safe_mean(g, "speed_spike_mean_per_k"),
                      "mean_reconnection": _safe_mean(g, "reconnection_rate")}
    out = {"n_windows": len(windows), "has_ground_truth": bool(has_gt),
           "per_method": agg,
           "note": ("geometry_3d/tracklet_bind/motion_resolver/hybrid share the 3D backbone; "
                    "they differ in unknown_frac (where they admit uncertainty), not in identity "
                    "labels. Only appearance_idtracker gives independent labels. Accuracy at "
                    "crossings is UNDECIDABLE until ground truth exists — see annotation_worklist.csv.")}
    (out_dir / "summary.json").write_text(json.dumps(_json_safe(out), indent=2), encoding="utf-8")


def _safe_mean(g, col):
    if col not in g.columns:
        return None
    v = pd.to_numeric(g[col], errors="coerce").dropna()
    return round(float(v.mean()), 4) if len(v) else None


def _write_worklist(windows, resolved, out_dir, gt):
    """Prioritised annotation worklist: where to label, what type, how many, why."""
    rows = []
    labeled_centers = set()
    if gt is not None and "center_frame" in gt.columns and "swap" in gt.columns:
        labeled_centers = set(gt.dropna(subset=["swap"])["center_frame"].astype(int))
    for w in windows:
        if w["kind"] == "suspect_swap":
            rows.append({"priority": 1, "window": w["name"], "frames": f"{w['start']}-{w['end']}",
                         "label_type": "fish-ID at crossing (which slot is which after the event)",
                         "n_examples": 1, "already_labeled": int(w["center"]) in labeled_centers,
                         "discriminates": "motion_resolver vs geometry_3d (resolver claims a real swap)",
                         "why": "velocity contradicts the linker -> highest chance of a real, fixable error"})
        elif w["kind"] == "dwell_ambiguous":
            rows.append({"priority": 2, "window": w["name"], "frames": f"{w['start']}-{w['end']}",
                         "label_type": "crossing-vs-mix + fish-ID through the dwell",
                         "n_examples": 1, "already_labeled": int(w["center"]) in labeled_centers,
                         "discriminates": "all geometry methods (none can resolve the hard core)",
                         "why": "long close-contact dwell: tests whether ANY method recovers identity"})
        elif w["kind"] == "early_fp":
            rows.append({"priority": 3, "window": w["name"], "frames": f"{w['start']}-{w['end']}",
                         "label_type": "false-positive flag + true fish count per frame",
                         "n_examples": 20, "already_labeled": False,
                         "discriminates": "FP rejection (geometry vs tracklet gating)",
                         "why": "single->three transition + reflections; tests FP handling"})
    # also surface the full suspect_swap population count for sizing
    n_suspect = int((resolved.get("reason") == "suspect_swap").sum()) if "reason" in resolved.columns else 0
    n_amb_long = int(((resolved.get("reason") == "ambiguous") & (resolved.get("n_frames", 0) >= 166)).sum()) \
        if "reason" in resolved.columns else 0
    df = pd.DataFrame(rows).sort_values("priority") if rows else pd.DataFrame()
    df.to_csv(out_dir / "annotation_worklist.csv", index=False)
    LOG.info("annotation worklist: %d windows queued; population: %d suspect_swap, %d long-ambiguous events",
             len(df), n_suspect, n_amb_long)


if __name__ == "__main__":
    main()
