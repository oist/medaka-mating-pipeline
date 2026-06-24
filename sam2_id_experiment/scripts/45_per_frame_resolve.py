from __future__ import annotations

"""Stage 5 — per-frame appearance resolution of the whole 3-fish track (production Line B).

Where the window-anchor female call (script 44) leaves ~46% of fish-frames low-confidence (geometry
propagation between sparse window anchors), this runs the clean classifiers PER FRAME over the whole
timeline and resolves every swap: female slot per frame (argmax P(female)), then M1/M2 among the two
non-female slots (argmax P(M2)), optionally fused with a TOP-view model where the sides merge. Output is
``<tag>_3d_individual_id.csv`` with ``individual_id in {F, M1, M2}`` + per-row confidence.

Needs FULL-LENGTH Detectron2 masks (the saion fan-out: ``<view>.masks_<S>-<E>.h5`` under --masks-dir)
and per-view slot/tracklet points (e.g. stage-0 ``slot_points`` of 40, or any frame,x,y,global_id CSV).
All resolution logic lives in ``per_frame_resolve`` (unit-tested, shared with the research scripts).

Run::

    python -m sam2_id_experiment.scripts.45_per_frame_resolve 20251225_K_1_2_FCMC \
        --paths-config configs/paths.local.toml \
        --masks-dir Z:/StephensU/medaka/detectron2_output/20251225_K_1_2_FCMC \
        --slot-points-dir runs/female_anchor/20251225_K_1_2_FCMC/round5_260611/slot_points \
        --female-model .../model_female_clean/model.pt --male-model .../model_male_clean/model.pt \
        [--top-model .../model_top_male/model.pt --top-view top_cam]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.common.local_paths import load_local_paths
from sam2_id_experiment.src.config import VIEW_VIDEO_BASENAMES
from sam2_id_experiment.src.id_model import per_frame_resolve as pfr
from sam2_id_experiment.src.id_model.reproject_slot_points import slot_from_fish_id

REPO = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Per-frame appearance resolution of the whole 3-fish track.")
    p.add_argument("experiment")
    p.add_argument("--paths-config", default="configs/paths.local.toml")
    p.add_argument("--round-tag", default="round5_260611")
    p.add_argument("--identified-3d-csv", default=None)
    p.add_argument("--masks-dir", required=True, help="full-length Detectron2 masks (<view>.masks_<S>-<E>.h5)")
    p.add_argument("--slot-points-dir", required=True, help="dir of <view>.tracklet_bound.csv (frame,x,y,global_id)")
    p.add_argument("--views", nargs="+", default=["side1_cam", "side2_cam"])
    p.add_argument("--female-model", required=True)
    p.add_argument("--male-model", required=True)
    p.add_argument("--top-model", default=None, help="optional top-view male model -> fuse to shrink the male-low core")
    p.add_argument("--top-view", default="top_cam")
    p.add_argument("--video", action="append", default=[], metavar="VIEW=PATH",
                   help="override a view's video path (else <medaka>/<exp>/<view>/<basename>); repeatable")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--crop-size", type=int, default=128)
    p.add_argument("--gate", type=float, default=40.0)
    p.add_argument("--margin-thr", type=float, default=0.30)
    p.add_argument("--high-dist", type=int, default=25)
    p.add_argument("--out-csv", default=None)
    p.add_argument("--device", default="cuda")
    return p


def _videos(args, views, medaka_root) -> dict:
    overrides = dict(v.split("=", 1) for v in args.video)
    out = {}
    for v in views:
        out[v] = Path(overrides[v]) if v in overrides else (
            Path(medaka_root) / args.experiment / v / VIEW_VIDEO_BASENAMES[v])
    return out


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    local = load_local_paths(args.paths_config)
    recon_root = Path(local.reconstruction_output_dir)
    medaka_root = Path(local.medaka_data_dir)
    identified_csv = Path(args.identified_3d_csv) if args.identified_3d_csv else (
        recon_root / args.round_tag / args.experiment / f"{args.round_tag}_3d_identified.csv")
    if not identified_csv.exists():
        raise SystemExit(f"identified 3D CSV not found: {identified_csv}")

    import torch
    device = args.device if torch.cuda.is_available() else "cpu"
    slot_dir = Path(args.slot_points_dir)
    slot_csvs = {v: slot_dir / f"{v}.tracklet_bound.csv" for v in args.views}
    videos = _videos(args, args.views, medaka_root)

    track = pd.read_csv(identified_csv)
    track["frame"] = track["frame"].astype(int)
    track["geometry_slot"] = track["fish_id"].map(slot_from_fish_id).astype(int)
    if not track["geometry_slot"].between(0, 2).all():
        bad = sorted(track.loc[~track.geometry_slot.between(0, 2), "fish_id"].unique())
        raise SystemExit(f"identified CSV has fish_id mapping outside geometry slots 0..2: {bad} "
                         "(expected exactly ID_1/ID_2/ID_3 for the 3-fish track)")
    n_frames = int(track["frame"].max()) + 1
    end = args.end if args.end is not None else n_frames - 1

    def fwd(model, crops):
        return pfr.forward_head(model, crops, device)

    common = dict(start=args.start, end=end, stride=args.stride, crop_size=args.crop_size, gate=args.gate)

    print(f"[female] per-frame P(female) over {args.views} ...", flush=True)
    fclf = pfr.load_classifier(args.female_model, device)
    female_p = pfr.per_frame_class_prob(args.views, args.masks_dir, videos, slot_csvs, fclf, fwd,
                                        prob_prefix="p", **common)
    fres = pfr.resolve_female_track(female_p, n_frames, args.margin_thr, args.high_dist)
    female_ff = fres["female_ff"]
    if len(fres["anchor_frames"]) == 0:
        raise SystemExit(
            "no confident female anchors over the requested range — the female model produced no "
            "frame with margin > --margin-thr (check masks/video alignment, the model, or lower --margin-thr)")

    print(f"[male] per-frame P(M2) over {args.views} ...", flush=True)
    mclf = pfr.load_classifier(args.male_model, device)
    male_p = pfr.per_frame_class_prob(args.views, args.masks_dir, videos, slot_csvs, mclf, fwd,
                                      prob_prefix="pb", **common)
    mres = pfr.resolve_male_track(male_p, female_ff, n_frames, args.margin_thr, args.high_dist)
    beta_final, male_conf = mres["beta_final"], mres["male_conf"]

    if args.top_model:
        print(f"[top] fusing top-view P(M2) ({args.top_view}) ...", flush=True)
        tclf = pfr.load_classifier(args.top_model, device)
        top_videos = _videos(args, [args.top_view], medaka_root)
        top_slot = {args.top_view: slot_dir / f"{args.top_view}.tracklet_bound.csv"}
        top_p = pfr.per_frame_class_prob([args.top_view], args.masks_dir, top_videos, top_slot, tclf, fwd,
                                         prob_prefix="tm", **common)
        taf, tabeta = pfr.top_male_anchors(top_p, female_ff, args.margin_thr)
        beta_final, male_conf = pfr.fuse_side_top(beta_final, male_conf, taf, tabeta, female_ff,
                                                  n_frames, args.high_dist)

    frames = track.frame.to_numpy()
    slots = track.geometry_slot.to_numpy().astype(int)
    track["app_female_slot"] = female_ff[frames]
    track["app_conf"] = np.asarray(fres["app_conf"])[frames]
    track["individual_id"] = pfr.assemble_individual_labels(frames, slots, female_ff, beta_final, n_frames)
    track["app_male_beta_slot"] = beta_final[frames]
    track["male_conf"] = np.asarray(male_conf)[frames]
    track["indiv_conf"] = np.where(track.individual_id == "F", track.app_conf, track.male_conf)

    out_csv = Path(args.out_csv) if args.out_csv else identified_csv.with_name(
        identified_csv.stem + "_individual_id.csv")
    track.to_csv(out_csv, index=False)

    counts = track.groupby("frame").individual_id.value_counts().unstack(fill_value=0)
    one_each = int(((counts.get("F", 0) == 1) & (counts.get("M1", 0) == 1) & (counts.get("M2", 0) == 1)).sum())
    per_fish = {fid: round(float((g.indiv_conf == "high").mean()), 3)
                for fid, g in track.groupby("individual_id")}
    report = {"experiment": args.experiment, "identified_3d_csv": str(identified_csv),
              "out_csv": str(out_csv), "n_frames": int(track.frame.nunique()),
              "fused_top": bool(args.top_model), "frames_one_of_each": one_each,
              "high_conf_fraction_by_individual": per_fish,
              "indiv_conf_counts": {k: int(v) for k, v in track.indiv_conf.value_counts().items()}}
    (out_csv.with_suffix(".report.json")).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n" + "=" * 60)
    print(f"wrote {out_csv}")
    print(f"  frames with exactly 1 F+1 M1+1 M2: {one_each}/{track.frame.nunique()}")
    print(f"  high-confidence by individual: {per_fish}")
    print("=" * 60)


if __name__ == "__main__":
    main()
