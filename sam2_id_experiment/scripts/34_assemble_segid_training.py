from __future__ import annotations

"""Stage B-prep: pool per-view seg-ID crops into one contrastive training set.

Reads the ``<view>_crops.h5`` + ``<view>_crop_metadata.parquet`` written by
``33_build_segid_dataset`` for each view, concatenates the crops into a single
global index, and builds self-supervised pairs with the STRICT same-frame sampler:

  * must-link  -- crops in the same (view, tracklet),
  * cannot-link -- crops in the same frame AND same view, different tracklet.

Crucially the 3D ``global_id`` (the geometry's per-tracklet fish slot, present in the
tracklet-bound CSV) is NOT used to build self-supervised pairs -- it is copied into
the metadata only as a HELD-OUT evaluation / optional weak-supervision label, so the
trained embedding can be scored on "does appearance separate the 3 fish"
independently of geometry. The modal label purity is also written because mixed
tracklets are label-noisy and should be excluded from supervised classifier runs.

Writes a dataset dir compatible with ``id_model.train.train_embedding_model``:
``crops.h5`` (dataset ``crops``), ``pairs.parquet``, ``crop_metadata.parquet``.

Run::

    python -m sam2_id_experiment.scripts.34_assemble_segid_training \
        --segid-dir runs/segid/<exp> --views top_cam side1_cam side2_cam \
        --tracklet-bound-dir <dir with *.tracklet_bound.csv> \
        --out-dir runs/segid/<exp>/pooled_dataset
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from sam2_id_experiment.src.id_dataset.pair_sampler import sample_constrained_pairs


def _global_id_stats(tb_csv: str) -> dict[int, dict[str, float | int]]:
    """tracklet_id -> modal 3D global_id plus modal-label purity."""
    if not Path(tb_csv).exists():
        return {}
    d = pd.read_csv(tb_csv, usecols=["tracklet_id", "global_id"])
    stats: dict[int, dict[str, float | int]] = {}
    for tracklet_id, labels in d.groupby("tracklet_id")["global_id"]:
        labels = labels.dropna()
        if labels.empty:
            continue
        counts = labels.value_counts()
        modal = counts.index[0]
        modal_n = int(counts.iloc[0])
        total = int(counts.sum())
        stats[int(tracklet_id)] = {
            "global_id": int(modal),
            "global_id_purity": float(modal_n / max(total, 1)),
            "global_id_modal_n": modal_n,
            "global_id_n": total,
        }
    return stats


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pool per-view seg-ID crops + build constrained pairs.")
    p.add_argument("--segid-dir", required=True, nargs="+",
                   help="one or more per-window seg-ID dirs to pool")
    p.add_argument("--views", nargs="+", default=["top_cam", "side1_cam", "side2_cam"])
    p.add_argument("--tracklet-bound-dir", default=None, help="for held-out global_id eval label")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--positive-pairs-per-tracklet", type=int, default=400)
    p.add_argument("--negative-pairs-per-frame", type=int, default=6)
    p.add_argument("--max-negative-pairs", type=int, default=60000)
    p.add_argument("--max-area-ratio", type=float, default=2.0)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    segid_dirs = [Path(d) for d in args.segid_dir]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gid_cache = {}
    all_images, all_meta, all_pairs = [], [], []
    offset = 0
    # pair sampling is per (window, view): cannot-link is same frame within one
    # window+view; tracklet_ids are globally unique per view so view_tracklet stays
    # consistent for the same fish-tracklet across windows.
    for di, segid in enumerate(segid_dirs):
        for vi, view in enumerate(args.views):
            h5p = segid / f"{view}_crops.h5"
            metap = segid / f"{view}_crop_metadata.parquet"
            if not h5p.exists() or not metap.exists():
                continue
            with h5py.File(h5p, "r") as h:
                imgs = h["images"][()]
            meta = pd.read_parquet(metap).sort_values("crop_id").reset_index(drop=True)
            assert len(meta) == len(imgs), f"{view}: meta/imgs mismatch"
            meta["local_crop_id"] = meta["crop_id"]
            meta["crop_id"] = np.arange(len(meta)) + offset
            meta["view"] = view
            meta["window"] = segid.name
            meta["view_tracklet"] = vi * 1_000_000 + meta["tracklet_id"].astype(int)
            if args.tracklet_bound_dir:
                if view not in gid_cache:
                    gid_cache[view] = _global_id_stats(
                        str(Path(args.tracklet_bound_dir) / f"{view}.tracklet_bound.csv"))
                gid_stats = gid_cache[view]
                meta["global_id"] = (
                    meta["tracklet_id"].map({k: v["global_id"] for k, v in gid_stats.items()})
                    .fillna(-1)
                    .astype(int)
                )
                meta["global_id_purity"] = (
                    meta["tracklet_id"].map({k: v["global_id_purity"] for k, v in gid_stats.items()})
                    .fillna(0.0)
                    .astype(float)
                )
                meta["global_id_modal_n"] = (
                    meta["tracklet_id"].map({k: v["global_id_modal_n"] for k, v in gid_stats.items()})
                    .fillna(0)
                    .astype(int)
                )
                meta["global_id_n"] = (
                    meta["tracklet_id"].map({k: v["global_id_n"] for k, v in gid_stats.items()})
                    .fillna(0)
                    .astype(int)
                )
            else:
                meta["global_id"] = -1
                meta["global_id_purity"] = 0.0
                meta["global_id_modal_n"] = 0
                meta["global_id_n"] = 0
            pair_meta = meta.rename(columns={"tracklet_id": "_orig_tid"})
            pair_meta["tracklet_id"] = pair_meta["view_tracklet"]
            pairs = sample_constrained_pairs(pair_meta, {
                "positive_pairs_per_tracklet": args.positive_pairs_per_tracklet,
                "negative_pairs_per_frame": args.negative_pairs_per_frame,
                "max_negative_pairs": args.max_negative_pairs,
                "max_area_ratio": args.max_area_ratio,
                "random_seed": 13 + di * 7 + vi,
            })
            all_images.append(imgs)
            all_meta.append(meta)
            all_pairs.append(pairs)
            offset += len(meta)
            print(f"{segid.name}/{view}: {len(meta)} crops, "
                  f"pos={int((pairs.label==1).sum())} neg={int((pairs.label==0).sum())}", flush=True)

    if not all_images:
        raise SystemExit("no view data found")
    images = np.concatenate(all_images, axis=0)
    meta = pd.concat(all_meta, ignore_index=True)
    pairs = pd.concat(all_pairs, ignore_index=True).sample(frac=1.0, random_state=7).reset_index(drop=True)

    with h5py.File(out_dir / "crops.h5", "w") as h:
        h.create_dataset("crops", data=images, compression="gzip")
    meta.to_parquet(out_dir / "crop_metadata.parquet", index=False)
    pairs.to_parquet(out_dir / "pairs.parquet", index=False)

    summary = {
        "n_crops": int(len(meta)), "views": args.views,
        "clean_train": int((meta.qc_status == "clean_train").sum()),
        "n_view_tracklets": int(meta.view_tracklet.nunique()),
        "pairs_positive": int((pairs.label == 1).sum()),
        "pairs_negative": int((pairs.label == 0).sum()),
        "global_id_labeled_crops": int((meta.global_id >= 0).sum()),
        "global_id_high_purity_crops": int(((meta.global_id >= 0) & (meta.global_id_purity >= 0.9)).sum()),
        "global_id_purity_quantiles": {
            str(q): float(meta.loc[meta.global_id >= 0, "global_id_purity"].quantile(q))
            for q in [0.0, 0.1, 0.5, 0.9, 1.0]
        } if (meta.global_id >= 0).any() else {},
    }
    (out_dir / "assemble_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\npooled -> {out_dir}\n{json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
