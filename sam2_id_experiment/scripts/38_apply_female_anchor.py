from __future__ import annotations

"""Apply the female anchor: tag each 3D fish identity F/M and (optionally) write a
sexed 3D-identified CSV. Production filter for the identity pipeline.

Per crop -> P(female) from the trained classifier; aggregate per geometry slot
(``global_id``) over ALL the fish's crops (robust to per-crop drift); label the
argmax slot female (exactly one), the rest male; flag the call if the top-vs-second
margin is small. With ``--identified-3d-csv`` it maps slot k -> ``ID_{k+1}`` -> adds
``sex``/``female_prob`` columns keyed by the geometry ``fish_id`` so downstream
mate-choice analysis can pick out the female (the chooser).

Run::

    python -m sam2_id_experiment.scripts.38_apply_female_anchor \
        --dataset <pooled_sides_full> --model <model_female/model.pt> \
        --out-dir <female_anchor> \
        [--identified-3d-csv <..._3d_identified.csv>]
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from sam2_id_experiment.src.id_model.female_anchor import (
    aggregate_female_prob, assign_single_female, predict_female_prob, sex_labels,
)
from sam2_id_experiment.src.id_model.resnet_embedder import EmbeddingClassifier, ResNet18Embedder


def _load_model(model_path, device):
    ck = torch.load(model_path, map_location=device)
    model = EmbeddingClassifier(ResNet18Embedder(embedding_dim=int(ck["embedding_dim"]),
                                                 pretrained=False), 2)
    model.load_state_dict(ck["model_state"]); model.to(device).eval()
    return model


def build_parser():
    p = argparse.ArgumentParser(description="Apply the female anchor to tag fish sex.")
    p.add_argument("--dataset", required=True, help="pooled crops dir (crops.h5 + crop_metadata.parquet)")
    p.add_argument("--model", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--identified-3d-csv", default=None)
    p.add_argument("--margin-threshold", type=float, default=0.15)
    p.add_argument("--device", default="cuda")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    with h5py.File(Path(args.dataset) / "crops.h5", "r") as h:
        crops = h["crops"][...]
    meta = pd.read_parquet(Path(args.dataset) / "crop_metadata.parquet").reset_index(drop=True)
    use = meta[(meta.qc_status == "clean_train") & (meta.global_id >= 0)].copy()

    model = _load_model(args.model, args.device)
    use["female_prob"] = predict_female_prob(model, crops[use.index.to_numpy()], args.device)

    # per-slot (geometry identity) aggregation -> single female
    slot_agg = aggregate_female_prob(use, "global_id").set_index("global_id")
    per_fish = slot_agg["mean_female_prob"].to_dict()
    n_crops = slot_agg["n_crops"].to_dict()
    call = assign_single_female(per_fish, n_crops, margin_threshold=args.margin_threshold)
    labels = sex_labels(per_fish, call.female_fish)

    slot_out = slot_agg.reset_index()
    slot_out["sex"] = slot_out.global_id.map(labels)
    slot_out["fish_id"] = slot_out.global_id.map(lambda k: f"ID_{int(k)+1}")
    slot_out.to_csv(out / "sex_assignment_by_slot.csv", index=False)

    # per-tracklet female prob (for QC / drift inspection)
    trk = aggregate_female_prob(use, "view_tracklet")
    trk["global_id"] = trk.view_tracklet.map(use.groupby("view_tracklet").global_id.first())
    trk.to_csv(out / "female_prob_by_tracklet.csv", index=False)

    report = {**asdict(call),
              "female_fish_slot": int(call.female_fish) if call.female_fish is not None else None,
              "female_fish_id": f"ID_{int(call.female_fish)+1}" if call.female_fish is not None else None,
              "sex_by_slot": {int(k): v for k, v in labels.items()},
              "margin_threshold": args.margin_threshold,
              "note": ("females vs males only; the two males are NOT distinguished here "
                       "(use 3D geometry + motion resolver for male1/male2).")}
    (out / "female_anchor_report.json").write_text(json.dumps(report, indent=2, default=str),
                                                   encoding="utf-8")

    # optional: merge sex into the 3D-identified CSV by geometry fish_id
    if args.identified_3d_csv and Path(args.identified_3d_csv).exists():
        df = pd.read_csv(args.identified_3d_csv)
        fishid_sex = {f"ID_{int(k)+1}": v for k, v in labels.items()}
        fishid_prob = {f"ID_{int(k)+1}": float(per_fish[k]) for k in per_fish}
        df["sex"] = df["fish_id"].map(fishid_sex)
        df["female_prob"] = df["fish_id"].map(fishid_prob)
        dst = Path(args.identified_3d_csv)
        outcsv = out / (dst.stem + "_sexed.csv")
        df.to_csv(outcsv, index=False)
        report["sexed_csv"] = str(outcsv)
        report["sexed_rows"] = int(len(df))
        (out / "female_anchor_report.json").write_text(json.dumps(report, indent=2, default=str),
                                                       encoding="utf-8")
        print(f"wrote sexed 3D CSV -> {outcsv}")

    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
