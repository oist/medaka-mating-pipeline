from __future__ import annotations

"""Classify each concrete tracklet into a fish ID or mix/crossing/unknown.

Goal #3 of the identity work: turn the per-view concrete tracklets into LABELLED
segments. The binder (`bind_tracklets_3d`) always forces a tracklet to its modal 3D
slot and only sets a `low_conf` flag; it has no explicit *mix / crossing / unknown*
class, so a tracklet that actually straddles two fish (a merged crossing) is
silently assigned to one. This module adds that classification.

Evidence per tracklet (re-derived from the 3D backbone, no .slp reload):
  * per-frame best 3D slot by reprojection of the 3 known slots into the view
    (the same geometry the binder validated at ~6 px residual / ~200 px margin),
  * reprojection residual and 2nd-best margin per frame,
  * fraction of frames inside a hard-core 3D event,
  * DYNAMICS: cosine consistency between the tracklet's 2D velocity and the modal
    slot's reprojected 2D velocity (a temporal cue the per-frame residual lacks),
  * optional APPEARANCE vote (contrastive-embedding id), fused only when asked --
    it is a moderate cue (per-crop ~44% direct, ~72% only after GT label-remap), so
    by default it is recorded for inspection but does NOT override geometry.

Decision (see ``classify_segment``):
  * ``mix``      -- the per-frame slot sequence is temporally segregated into >=2
                    sustained runs of different slots (the tracklet contains a
                    crossing/merge of two fish);
  * ``crossing`` -- mostly inside a hard-core event and not cleanly one slot;
  * ``ID_k``     -- one slot dominates with high consistency, margin and low
                    residual (and, if enabled, passes the dynamics / appearance veto);
  * ``unknown``  -- ambiguous single slot (thin margin / high residual / motion
                    inconsistent).

Run::

    python -m scripts.tracking.classify_segments \
        --csv-3d <..._3d_identified.csv> --bind-dir <tracklet_bind/> \
        --out-dir runs/segment_class/<exp>
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.common.local_paths import choose_path, load_local_paths
from scripts.reconstruction_3d.calibration import Calibration
from scripts.tracking.bind_tracklets_3d import (
    VIEWS, SLOTS, pairwise_3d, detect_hardcore_events, _reproject_all,
)


# --------------------------- pure, testable logic --------------------------- #

def run_length(seq) -> list[tuple[int, int]]:
    """[(value, length), ...] for consecutive runs in ``seq``."""
    out = []
    for v in seq:
        if out and out[-1][0] == v:
            out[-1] = (v, out[-1][1] + 1)
        else:
            out.append((int(v), 1))
    return out


def velocity_cosine(track_xy: np.ndarray, slot_xy: np.ndarray) -> float:
    """Mean cosine similarity between consecutive-frame displacement of the tracklet
    and of the modal slot's reprojection. NaN if too few finite steps. This rewards
    a segment that *moves like* the 3D fish it is assigned to (dynamics, not just
    instantaneous proximity)."""
    if len(track_xy) < 3:
        return float("nan")
    dt = np.diff(track_xy, axis=0)
    ds = np.diff(slot_xy, axis=0)
    nt = np.linalg.norm(dt, axis=1)
    ns = np.linalg.norm(ds, axis=1)
    ok = np.isfinite(nt) & np.isfinite(ns) & (nt > 1e-6) & (ns > 1e-6)
    if ok.sum() < 2:
        return float("nan")
    cos = np.sum(dt[ok] * ds[ok], axis=1) / (nt[ok] * ns[ok])
    return float(np.mean(cos))


def classify_segment(slots, margins, resids, in_hc, *,
                     motion_consistency=float("nan"),
                     app_id=None, app_purity=float("nan"),
                     consistency_min=0.9, margin_min=10.0, resid_max=20.0,
                     hc_frac_max=0.5, mix_p2_min=0.3, min_run=5,
                     motion_min=-0.2, use_appearance=False) -> dict:
    """Classify one tracklet from its per-frame arrays. Returns a record with the
    label, confidence, the modal slot and all decision features. Pure."""
    slots = np.asarray(slots, dtype=int)
    n = len(slots)
    base = {"label": "unknown", "confidence": 0.0, "modal_slot": -1, "second_slot": -1,
            "p1": 0.0, "p2": 0.0, "n_frames": int(n), "median_margin": float("nan"),
            "mean_resid": float("nan"), "hardcore_frac": float("nan"),
            "motion_consistency": float(motion_consistency), "n_runs": 0,
            "app_id": (int(app_id) if app_id is not None else -1),
            "app_purity": float(app_purity), "reason": "empty"}
    if n == 0:
        return base
    counts = np.bincount(slots, minlength=len(SLOTS))
    order = np.argsort(counts)[::-1]
    modal, second = int(order[0]), int(order[1]) if len(order) > 1 else -1
    p1 = float(counts[modal] / n)
    p2 = float(counts[second] / n) if second >= 0 else 0.0
    hc_frac = float(np.mean(in_hc)) if len(in_hc) else 0.0
    fin_m = np.asarray(margins, dtype=float)
    fin_r = np.asarray(resids, dtype=float)
    med_margin = float(np.nanmedian(fin_m)) if np.isfinite(fin_m).any() else float("nan")
    mean_resid = float(np.nanmean(fin_r)) if np.isfinite(fin_r).any() else float("nan")

    runs = run_length(slots)
    long_runs = [r for r in runs if r[1] >= min_run]
    distinct_long = {r[0] for r in long_runs}

    rec = {**base, "modal_slot": modal, "second_slot": second, "p1": round(p1, 4),
           "p2": round(p2, 4), "median_margin": med_margin, "mean_resid": mean_resid,
           "hardcore_frac": round(hc_frac, 4), "n_runs": len(runs)}

    # 1) mix: the segment is temporally split between >=2 fish (a crossing merge)
    if len(distinct_long) >= 2 and p2 >= mix_p2_min:
        rec.update(label="mix", confidence=round(p1 + p2, 4), reason="segregated_runs")
        return rec
    # 2) crossing: mostly inside a hard-core event and not cleanly one slot
    if hc_frac > hc_frac_max and p1 < consistency_min:
        rec.update(label="crossing", confidence=round(1 - p1, 4), reason="in_hardcore")
        return rec
    # 3) confident single ID
    if p1 >= consistency_min and (not np.isfinite(med_margin) or med_margin >= margin_min) \
            and (not np.isfinite(mean_resid) or mean_resid <= resid_max):
        # dynamics veto
        if np.isfinite(motion_consistency) and motion_consistency < motion_min:
            rec.update(label="unknown", confidence=round(p1, 4), reason="motion_inconsistent")
            return rec
        # optional appearance veto (off by default: appearance is only moderate)
        reason = "confident_single"
        if use_appearance and app_id is not None and app_id >= 0 and np.isfinite(app_purity):
            if app_id != modal and app_purity >= 0.6:
                rec.update(label="unknown", confidence=round(p1, 4),
                           reason="appearance_conflict")
                return rec
            if app_id == modal:
                reason = "confident_single+appearance"
        rec.update(label=f"ID_{modal}", confidence=round(p1, 4), reason=reason)
        return rec
    # 4) ambiguous single
    rec.update(label="unknown", confidence=round(p1, 4), reason="thin_margin_or_resid")
    return rec


# ------------------------------- orchestration ------------------------------ #

def _extract_tracklet_arrays(g, reproj_view, frame_to_idx, hardcore_frames):
    g = g.sort_values("frame")
    slots, margins, resids, in_hc, txy, sidx = [], [], [], [], [], []
    for f, x, y in zip(g["frame"].to_numpy(), g["x"].to_numpy(), g["y"].to_numpy()):
        i = frame_to_idx.get(int(f))
        if i is None:
            continue
        d = np.linalg.norm(reproj_view[:, i, :] - np.array([x, y]), axis=1)
        fin = np.isfinite(d)
        if fin.sum() < 1:
            continue
        order = np.argsort(np.where(fin, d, np.inf))
        best = int(order[0])
        slots.append(best)
        resids.append(float(d[best]))
        margins.append(float(d[order[1]] - d[order[0]]) if fin.sum() >= 2 else np.nan)
        in_hc.append(int(f) in hardcore_frames)
        txy.append((x, y))
        sidx.append(i)
    return (np.array(slots), np.array(margins), np.array(resids),
            np.array(in_hc), np.array(txy, dtype=float), np.array(sidx))


def classify_view(df, reproj_view, frame_to_idx, hardcore_frames, app_map, th):
    rows = []
    for tid, g in df.groupby("tracklet_id"):
        slots, margins, resids, in_hc, txy, sidx = _extract_tracklet_arrays(
            g, reproj_view, frame_to_idx, hardcore_frames)
        if len(slots) == 0:
            continue
        modal = int(np.bincount(slots, minlength=len(SLOTS)).argmax())
        mc = velocity_cosine(txy, reproj_view[modal, sidx, :]) if len(sidx) else float("nan")
        app = app_map.get(int(tid), {})
        rec = classify_segment(
            slots, margins, resids, in_hc, motion_consistency=mc,
            app_id=app.get("app_id"), app_purity=app.get("app_purity", float("nan")),
            **th)
        rec.update({"tracklet_id": int(tid), "start": int(g["frame"].min()),
                    "end": int(g["frame"].max()), "length": int(len(slots))})
        rows.append(rec)
    return rows


def _load_appearance(path):
    """Optional. CSV/parquet with columns [view, tracklet_id, app_id, app_purity].
    Returns {view: {tracklet_id: {app_id, app_purity}}}."""
    if not path or not Path(path).exists():
        return {}
    p = Path(path)
    d = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    if "tracklet_id" not in d.columns or "app_id" not in d.columns:
        return {}
    out: dict[str, dict] = {}
    for _, r in d.iterrows():
        v = str(r.get("view", "all"))
        out.setdefault(v, {})[int(r["tracklet_id"])] = {
            "app_id": int(r["app_id"]),
            "app_purity": float(r.get("app_purity", float("nan"))),
        }
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Classify tracklets into ID / mix / crossing / unknown.")
    p.add_argument("--csv-3d", required=True)
    p.add_argument("--bind-dir", required=True, help="dir with <view>.tracklet_bound.csv")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--model-folder", default=None)
    p.add_argument("--paths-config", default=None)
    p.add_argument("--hardcore-mm", type=float, default=12.0)
    p.add_argument("--event-merge-gap", type=int, default=5)
    p.add_argument("--consistency-min", type=float, default=0.9)
    p.add_argument("--margin-min", type=float, default=10.0)
    p.add_argument("--resid-max", type=float, default=20.0)
    p.add_argument("--hc-frac-max", type=float, default=0.5)
    p.add_argument("--mix-p2-min", type=float, default=0.3)
    p.add_argument("--min-run", type=int, default=5)
    p.add_argument("--motion-min", type=float, default=-0.2,
                   help="modal-slot velocity cosine below this vetoes an ID -> unknown")
    p.add_argument("--appearance-csv", default=None,
                   help="optional [view,tracklet_id,app_id,app_purity] to fuse")
    p.add_argument("--use-appearance", action="store_true",
                   help="let a confident appearance vote veto a geometry ID (default: record only)")
    p.add_argument("--min-len", type=int, default=5)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_folder = args.model_folder
    if model_folder is None:
        local = load_local_paths(args.paths_config)
        model_folder = choose_path(None, local.calibration_model_dir if local else None)
    if model_folder is None:
        raise SystemExit("need --model-folder or calibration_model_dir in paths.local.toml")
    calib = Calibration(str(model_folder))

    print(f"loading 3D {args.csv_3d} ...", flush=True)
    df3 = pd.read_csv(args.csv_3d)
    piv = df3.pivot_table(index="frame", columns="fish_id", values=["X", "Y", "Z"])
    frames = piv.index.to_numpy()
    frame_to_idx = {int(f): i for i, f in enumerate(frames)}
    xyz = {s: np.stack([piv[("X", s)].to_numpy(), piv[("Y", s)].to_numpy(),
                        piv[("Z", s)].to_numpy()], axis=1) for s in SLOTS}

    events = detect_hardcore_events(pairwise_3d(xyz), frames, hardcore_mm=args.hardcore_mm,
                                    merge_gap=args.event_merge_gap)
    hardcore_frames = set()
    for ev in events:
        hardcore_frames.update(range(ev["start"], ev["end"] + 1))
    proj = _reproject_all(calib, xyz)
    appearance = _load_appearance(args.appearance_csv)

    th = dict(consistency_min=args.consistency_min, margin_min=args.margin_min,
              resid_max=args.resid_max, hc_frac_max=args.hc_frac_max,
              mix_p2_min=args.mix_p2_min, min_run=args.min_run,
              motion_min=args.motion_min, use_appearance=args.use_appearance)

    summary = {"hardcore_mm": args.hardcore_mm, "n_events": len(events),
               "use_appearance": bool(args.use_appearance), "views": {}}
    for v in VIEWS:
        bound = Path(args.bind_dir) / f"{v}.tracklet_bound.csv"
        if not bound.exists():
            print(f"  skip {v}: no {bound.name}")
            continue
        d = pd.read_csv(bound)
        d = d[d.groupby("tracklet_id")["frame"].transform("size") >= args.min_len]
        app_map = appearance.get(v, appearance.get("all", {}))
        rows = classify_view(d, proj[v], frame_to_idx, hardcore_frames, app_map, th)
        rdf = pd.DataFrame(rows).sort_values("start")
        out_csv = out_dir / f"{v}.segment_class.csv"
        rdf.to_csv(out_csv, index=False)
        counts = Counter(rdf["label"])
        n = len(rdf)
        summary["views"][v] = {"n_tracklets": n, "label_counts": dict(counts),
                               "frac_id": round(sum(c for l, c in counts.items()
                                                    if l.startswith("ID_")) / max(n, 1), 4),
                               "frac_mix": round(counts.get("mix", 0) / max(n, 1), 4),
                               "frac_crossing": round(counts.get("crossing", 0) / max(n, 1), 4),
                               "frac_unknown": round(counts.get("unknown", 0) / max(n, 1), 4)}
        print(f"  {v}: {n} tracklets -> {dict(counts)}  -> {out_csv.name}")

    (out_dir / "segment_class_summary.json").write_text(json.dumps(summary, indent=2),
                                                        encoding="utf-8")
    print(f"\nwrote -> {out_dir}")


if __name__ == "__main__":
    main()
