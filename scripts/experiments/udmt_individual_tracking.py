from __future__ import annotations

"""Faithful test of UDMT-style individual tracking on Detectron2 segmentations.

Pipeline (single view):
  Detectron2 masks (raw, incl. reflections)
    -> mask centroids
    -> UDMT.tracking tracking-by-segmentation (motion-Hungarian, fwd + bwd)
    -> UDMT.bidirectional repair (forward/backward consensus) + crossing detection
    -> UDMT.metrics quality score
  Evaluated against PROXY GROUND TRUTH = the 3D `final_fish_id` reprojected into
  the view (the geometry identity is the best available per-frame label).

Reports MOT-style identity metrics (ID switches, per-track identity consistency,
coverage, fragments), forward-only vs bidirectional-repaired, and the female
(distinct fish) vs the two males split. This measures what individual ID UDMT can
recover from segmentation + motion alone -- the appearance ceiling is already
known (males unseparable), so this is the motion/segmentation ceiling.

Run::

    python -m scripts.experiments.udmt_individual_tracking \
      --view top_cam --masks <detectron2_masks.h5> \
      --identified-3d-csv <..._3d_identified.csv> --window 176300 181300 \
      --paths-config configs/paths.local.toml --out-dir runs/udmt_track/<exp>/<view>
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.common.local_paths import choose_path, load_local_paths
from scripts.reconstruction_3d.calibration import Calibration
from sam2_id_experiment.src.sam2_adapter.mask_io import iter_mask_instances_h5
from UDMT.bidirectional import BidirectionalConfig, CrossingConfig, detect_crossing_events, \
    repair_tracks_with_backward
from UDMT.metrics import TrackQualityConfig, score_track_table
from UDMT.tracking import TrackingConfig, track_forward_backward


# ------------------------------ data loading ------------------------------ #

def load_mask_centroids(masks_h5: str, start: int, end: int) -> pd.DataFrame:
    rows = []
    for inst in iter_mask_instances_h5(masks_h5):
        f = int(inst.frame)
        if not (start <= f <= end):
            continue
        m = np.asarray(inst.mask).astype(bool)
        ys, xs = np.nonzero(m)
        if len(xs) == 0:
            continue
        rows.append({"frame": f, "x": float(xs.mean()), "y": float(ys.mean()),
                     "area": int(len(xs)), "score": float(inst.score)})
    return pd.DataFrame(rows, columns=["frame", "x", "y", "area", "score"])


def reproject_gt(csv_3d: str, view: str, model_folder: str, start: int, end: int) -> pd.DataFrame:
    """3D final_fish_id reprojected into the view -> proxy ground truth
    (frame, fish, gx, gy). 15 = female (the cleanly separable fish)."""
    d = pd.read_csv(csv_3d)
    d = d[(d.frame >= start) & (d.frame <= end)].dropna(subset=["X", "Y", "Z"])
    calib = Calibration(str(model_folder))
    out = []
    for fish, g in d.groupby("final_fish_id"):
        xyz = g[["X", "Y", "Z"]].to_numpy(dtype=float)
        uv = calib.project_world_to_view(xyz, view)
        for (fr, _), (u, v) in zip(g[["frame", "final_fish_id"]].to_numpy(), uv):
            if np.isfinite(u) and np.isfinite(v):
                out.append({"frame": int(fr), "fish": int(fish), "gx": float(u), "gy": float(v)})
    return pd.DataFrame(out)


# ------------------------------ evaluation -------------------------------- #

def match_tracks_to_gt(tracks: pd.DataFrame, gt: pd.DataFrame, match_px: float) -> pd.DataFrame:
    """Per frame, Hungarian-match track centers to reprojected GT fish (gated).
    Returns rows (frame, track_id, fish, dist)."""
    rows = []
    gt_by_frame = {f: g for f, g in gt.groupby("frame")}
    for f, tg in tracks.groupby("frame"):
        gg = gt_by_frame.get(int(f))
        if gg is None or tg.empty:
            continue
        txy = tg[["x", "y"]].to_numpy(dtype=float)
        gxy = gg[["gx", "gy"]].to_numpy(dtype=float)
        cost = np.linalg.norm(txy[:, None, :] - gxy[None, :, :], axis=2)
        gated = np.where(cost > match_px, 1e9, cost)
        ri, ci = linear_sum_assignment(gated)
        for r, c in zip(ri, ci):
            if gated[r, c] >= 1e9:
                continue
            rows.append({"frame": int(f), "track_id": int(tg.iloc[r]["track_id"]),
                         "fish": int(gg.iloc[c]["fish"]), "dist": float(cost[r, c])})
    return pd.DataFrame(rows, columns=["frame", "track_id", "fish", "dist"])


def identity_metrics(tracks: pd.DataFrame, gt: pd.DataFrame, matches: pd.DataFrame,
                     female_fish: int) -> dict:
    n_window_frames = int(gt["frame"].nunique())
    n_gt_slots = int(len(gt))                       # frame x fish present
    n_fish = int(gt["fish"].nunique())

    # per-track global identity = modal matched fish; id switches = consecutive changes
    track_assign = {}
    total_switches = 0
    consistency_num = consistency_den = 0
    for tid, g in matches.sort_values("frame").groupby("track_id"):
        fish_seq = g["fish"].to_numpy()
        modal = int(pd.Series(fish_seq).mode().iloc[0])
        track_assign[int(tid)] = modal
        total_switches += int(np.sum(fish_seq[1:] != fish_seq[:-1]))
        consistency_num += int(np.sum(fish_seq == modal))
        consistency_den += len(fish_seq)

    # coverage: frac of GT slots that got a matched track
    covered = int(len(matches.drop_duplicates(["frame", "fish"])))
    coverage = covered / max(n_gt_slots, 1)

    # IDF1-ish: assign each GT fish its best track (Hungarian on overlap), then frac
    # of GT slots where that track is present and matched to the right fish.
    fish_ids = sorted(gt["fish"].unique())
    track_ids = sorted(matches["track_id"].unique())
    overlap = np.zeros((len(fish_ids), len(track_ids)))
    fidx = {f: i for i, f in enumerate(fish_ids)}
    tidx = {t: j for j, t in enumerate(track_ids)}
    for (tid, fish), grp in matches.groupby(["track_id", "fish"]):
        overlap[fidx[fish], tidx[tid]] += len(grp)
    idf1 = float("nan")
    fish_to_track = {}
    if len(track_ids):
        ri, ci = linear_sum_assignment(-overlap)
        correct = 0
        for r, c in zip(ri, ci):
            fish_to_track[fish_ids[r]] = track_ids[c]
            correct += int(overlap[r, c])
        idf1 = correct / max(n_gt_slots, 1)

    # per-fish (female vs males): id consistency of the fish's assigned best track
    per_fish = {}
    for fish in fish_ids:
        tid = fish_to_track.get(fish)
        sub = matches[matches.track_id == tid] if tid is not None else matches.iloc[0:0]
        on_fish = int((sub["fish"] == fish).sum())
        per_fish[int(fish)] = {
            "is_female": bool(fish == female_fish),
            "gt_frames": int((gt.fish == fish).sum()),
            "best_track": None if tid is None else int(tid),
            "frames_on_fish": on_fish,
            "id_consistency": round(on_fish / max(len(sub), 1), 4) if len(sub) else 0.0,
            "recall": round(on_fish / max(int((gt.fish == fish).sum()), 1), 4),
        }

    return {
        "n_window_frames": n_window_frames,
        "n_gt_slots": n_gt_slots,
        "n_fish": n_fish,
        "n_tracks": int(tracks["track_id"].nunique()),
        "n_tracks_matched": int(matches["track_id"].nunique()),
        "fragments_per_fish": round(tracks["track_id"].nunique() / max(n_fish, 1), 2),
        "total_id_switches": int(total_switches),
        "switches_per_1k_frames": round(total_switches * 1000 / max(n_window_frames, 1), 2),
        "track_identity_consistency": round(consistency_num / max(consistency_den, 1), 4),
        "coverage": round(coverage, 4),
        "idf1_like": round(idf1, 4) if np.isfinite(idf1) else None,
        "per_fish": per_fish,
        "track_global_assignment": {int(k): int(v) for k, v in track_assign.items()},
    }


def build_parser():
    p = argparse.ArgumentParser(description="UDMT-style individual tracking test on Detectron2 masks.")
    p.add_argument("--view", required=True)
    p.add_argument("--masks", required=True)
    p.add_argument("--identified-3d-csv", required=True)
    p.add_argument("--window", nargs=2, type=int, required=True, metavar=("START", "END"))
    p.add_argument("--out-dir", required=True)
    p.add_argument("--model-folder", default=None)
    p.add_argument("--paths-config", default=None)
    p.add_argument("--female-fish", type=int, default=15)
    p.add_argument("--num-individuals", type=int, default=3)
    p.add_argument("--max-jump-px", type=float, default=60.0)
    p.add_argument("--max-gap-frames", type=int, default=30)
    p.add_argument("--min-area", type=float, default=80.0)
    p.add_argument("--match-px", type=float, default=80.0)
    p.add_argument("--crossing-distance", type=float, default=35.0)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    start, end = args.window
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    model_folder = args.model_folder
    if model_folder is None:
        local = load_local_paths(args.paths_config)
        model_folder = choose_path(None, local.calibration_model_dir if local else None)
    if model_folder is None:
        raise SystemExit("need --model-folder or calibration_model_dir in paths.local.toml")

    print(f"[{args.view}] loading mask centroids {start}-{end} ...", flush=True)
    masks = load_mask_centroids(args.masks, start, end)
    masks.to_csv(out / "mask_centroids.csv", index=False)
    print(f"  {len(masks)} masks, {masks.frame.nunique()} frames, "
          f"{round(len(masks)/max(masks.frame.nunique(),1),2)} masks/frame", flush=True)

    tcfg = TrackingConfig(num_individuals=args.num_individuals, max_jump_px=args.max_jump_px,
                          max_gap_frames=args.max_gap_frames, min_area=args.min_area)
    fwd, bwd = track_forward_backward(masks, tcfg)
    fwd.to_csv(out / "forward.tracks.csv", index=False)
    bwd.to_csv(out / "backward.tracks.csv", index=False)

    repaired, rep_summary, votes = repair_tracks_with_backward(
        fwd, bwd, BidirectionalConfig(repair_crossings=True,
                                      crossing_distance_px=args.crossing_distance))
    repaired.to_csv(out / "repaired.tracks.csv", index=False)

    crossings = detect_crossing_events(fwd, CrossingConfig(
        id_column="track_id", coordinate_columns=("x", "y"),
        distance_threshold=args.crossing_distance, min_duration_frames=2, merge_gap_frames=10))
    crossings.to_csv(out / "crossings.csv", index=False)

    qcfg = TrackQualityConfig(num_individuals=args.num_individuals,
                              close_contact_distance_px=args.crossing_distance)
    q_fwd = score_track_table(fwd, qcfg).to_dict()
    q_rep = score_track_table(repaired, qcfg).to_dict()

    gt = reproject_gt(args.identified_3d_csv, args.view, model_folder, start, end)
    gt.to_csv(out / "gt_reprojected.csv", index=False)

    results = {}
    for name, trk in [("forward", fwd), ("repaired", repaired)]:
        matches = match_tracks_to_gt(trk[["frame", "track_id", "x", "y"]], gt, args.match_px)
        results[name] = identity_metrics(trk, gt, matches, args.female_fish)

    report = {
        "view": args.view, "window": [start, end],
        "tracking_config": tcfg.to_dict(),
        "n_masks": int(len(masks)),
        "masks_per_frame": round(len(masks) / max(masks.frame.nunique(), 1), 3),
        "n_crossings_detected": int(len(crossings)),
        "udmt_quality_forward": q_fwd,
        "udmt_quality_repaired": q_rep,
        "repair_summary": rep_summary.to_dict(),
        "identity_vs_3d": results,
        "note": ("proxy ground truth = 3D final_fish_id reprojected into the view; "
                 "tracker is motion-only (no appearance) -- the fair UDMT ceiling for "
                 "medaka since males carry no appearance signal."),
    }
    (out / "udmt_tracking_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _print_summary(report)


def _print_summary(r):
    print(f"\n=== {r['view']} {r['window']} ===")
    print(f"masks/frame {r['masks_per_frame']} | crossings {r['n_crossings_detected']}")
    for name in ("forward", "repaired"):
        m = r["identity_vs_3d"][name]
        print(f"  [{name}] tracks {m['n_tracks']} (frag/fish {m['fragments_per_fish']}) | "
              f"ID-switches {m['total_id_switches']} ({m['switches_per_1k_frames']}/1k) | "
              f"consistency {m['track_identity_consistency']} | coverage {m['coverage']} | "
              f"IDF1~ {m['idf1_like']}")
        fem = [v for v in m["per_fish"].values() if v["is_female"]]
        males = [v for v in m["per_fish"].values() if not v["is_female"]]
        if fem:
            print(f"      female recall {fem[0]['recall']} consistency {fem[0]['id_consistency']} | "
                  f"males recall {[mm['recall'] for mm in males]} consistency {[mm['id_consistency'] for mm in males]}")


if __name__ == "__main__":
    main()
