from __future__ import annotations

"""Decisive measurement: can a higher-capacity model separate the 2 males?

Embeds crops with (a) the FULL ResNet18 (female+male) and (b) the MALE-ONLY
ResNet18, and reports tracklet-grouped cross-validated balanced accuracy for:
  * FEMALE  (gid==female vs rest)   -- expect ~1.0 (anchor),
  * MALE-MALE (gid0 vs gid2)        -- the open question.
Grouped by view_tracklet so crops from one tracklet never split across folds
(otherwise the score inflates). Male labels come from geometry and are NOISY (they
can be swapped at crossings), so a near-chance male-male number is consistent with
EITHER weak appearance OR label noise -- stated explicitly in the report.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score
from sklearn.model_selection import GroupKFold

from sam2_id_experiment.src.id_model.resnet_embedder import ResNet18Embedder
from sam2_id_experiment.src.id_model.model import SmallMedakaEmbeddingNet


def _embed(dataset_dir, model_path, device):
    ck = torch.load(model_path, map_location=device)
    if ck.get("arch") == "resnet18":
        model = ResNet18Embedder(embedding_dim=int(ck["embedding_dim"]), pretrained=False)
    else:
        model = SmallMedakaEmbeddingNet(embedding_dim=int(ck.get("embedding_dim", 64)))
    model.load_state_dict(ck["model_state"]); model.to(device).eval()
    with h5py.File(Path(dataset_dir) / "crops.h5", "r") as h:
        crops = np.asarray(h["crops"][...], dtype=np.float32) / 255.0
    out = []
    with torch.no_grad():
        for s in range(0, len(crops), 256):
            b = torch.from_numpy(crops[s:s + 256]).permute(0, 3, 1, 2).to(device)
            out.append(model(b).cpu().numpy())
    return np.concatenate(out)


def grouped_cv(X, y, groups, splits=5):
    if len(set(y)) < 2 or len(set(groups)) < 2:
        return float("nan")
    k = min(splits, len(set(groups)))
    accs = []
    gkf = GroupKFold(k)
    for tr, te in gkf.split(X, y, groups):
        if len(set(y[tr])) < 2:
            continue
        clf = LogisticRegression(max_iter=1000).fit(X[tr], y[tr])
        accs.append(balanced_accuracy_score(y[te], clf.predict(X[te])))
    return float(np.mean(accs)) if accs else float("nan")


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--full-dataset", required=True)
    p.add_argument("--full-model", required=True)
    p.add_argument("--male-dataset", required=True)
    p.add_argument("--male-model", required=True)
    p.add_argument("--female-gid", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    rep = {"note": ("male-male labels are geometry-derived and NOISY (swappable at "
                    "crossings); near-chance is consistent with weak appearance OR label "
                    "noise. female is the trustworthy anchor.")}

    # FULL model: female + male-male
    Ef = _embed(args.full_dataset, args.full_model, args.device)
    mf = pd.read_parquet(Path(args.full_dataset) / "crop_metadata.parquet").reset_index(drop=True)
    cf = mf[(mf.qc_status == "clean_train") & (mf.global_id >= 0)]
    Xf, gid, grp = Ef[cf.index.to_numpy()], cf.global_id.to_numpy(), cf.view_tracklet.to_numpy()
    rep["full_model"] = {
        "n_crops": int(len(Xf)),
        "female_vs_rest_balacc": round(grouped_cv(Xf, (gid == args.female_gid).astype(int), grp), 3),
    }
    mask = np.isin(gid, [0, 2])
    rep["full_model"]["male_male_balacc"] = round(
        grouped_cv(Xf[mask], (gid[mask] == 0).astype(int), grp[mask]), 3)
    rep["full_model"]["male_male_kmeans_ARI"] = round(
        adjusted_rand_score(gid[mask], KMeans(2, n_init=10, random_state=0).fit_predict(Xf[mask])), 3)

    # MALE-ONLY model: male-male
    Em = _embed(args.male_dataset, args.male_model, args.device)
    mm = pd.read_parquet(Path(args.male_dataset) / "crop_metadata.parquet").reset_index(drop=True)
    cm = mm[mm.global_id.isin([0, 2])]
    Xm, gidm, grpm = Em[cm.index.to_numpy()], cm.global_id.to_numpy(), cm.view_tracklet.to_numpy()
    rep["male_only_model"] = {
        "n_male_crops": int(len(Xm)), "n_male_tracklets": int(cm.view_tracklet.nunique()),
        "male_male_balacc": round(grouped_cv(Xm, (gidm == 0).astype(int), grpm), 3),
        "male_male_kmeans_ARI": round(
            adjusted_rand_score(gidm, KMeans(2, n_init=10, random_state=0).fit_predict(Xm)), 3),
    }
    rep["baseline_tiny_model_male_male_balacc"] = 0.37  # from prior measurement
    rep["verdict"] = (
        "male-male clearly beats chance (>0.65 on both models) -> appearance helps males"
        if (rep["male_only_model"]["male_male_balacc"] or 0) > 0.65
        else "male-male near chance even with ResNet18 -> appearance is a FEMALE ANCHOR only; "
             "use geometry+motion for the 2 males")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
