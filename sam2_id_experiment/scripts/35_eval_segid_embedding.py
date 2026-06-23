from __future__ import annotations

"""Stage D: score the self-supervised embedding on "does appearance separate the
3 fish" -- honestly, using the held-out 3D ``global_id`` as the evaluation label.

The embedding was trained with NO identity labels (only same-tracklet / same-frame
constraints). Here we embed every clean crop, cluster, and compare the clustering
to the 3D-slot ``global_id`` (consistent across views) with metrics that are NOT
inflated by label-remapping:

  * silhouette(embeddings, global_id) -- separability of the true 3 groups
    (idtracker needs >=0.91 for <1% per-frame error; ~0.73 was "moderate"),
  * ARI(kmeans3, global_id)           -- permutation-invariant clustering agreement,
  * per-tracklet majority accuracy with Hungarian remap -- comparable to the prior
    44% direct / 72-77% remapped numbers,
  * cannot_link_satisfied              -- of same-view-same-frame crop pairs, the
    fraction the embedding places in different clusters (sanity on the supervision).

Run::

    python -m sam2_id_experiment.scripts.35_eval_segid_embedding \
        --dataset <pooled_dataset> --model <model.pt> --out-dir <eval_dir>
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sam2_id_experiment.src.id_model.model import SmallMedakaEmbeddingNet
from sam2_id_experiment.src.id_model.resnet_embedder import ResNet18Embedder


def embed(dataset_dir, model_path, device, batch_size=256):
    ckpt = torch.load(model_path, map_location=device)
    if ckpt.get("arch") == "resnet18":
        model = ResNet18Embedder(embedding_dim=int(ckpt.get("embedding_dim", 128)), pretrained=False).to(device)
    else:
        model = SmallMedakaEmbeddingNet(embedding_dim=int(ckpt.get("embedding_dim", 64))).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    with h5py.File(Path(dataset_dir) / "crops.h5", "r") as h:
        crops = np.asarray(h["crops"][...], dtype=np.float32) / 255.0
    embs = []
    with torch.no_grad():
        for s in range(0, len(crops), batch_size):
            b = torch.from_numpy(crops[s:s + batch_size]).permute(0, 3, 1, 2).to(device)
            embs.append(model(b).cpu().numpy())
    return np.concatenate(embs, axis=0)


def remap_accuracy(pred, true):
    """Hungarian best-match accuracy between arbitrary cluster ids and true labels."""
    labels_p = sorted(set(pred)); labels_t = sorted(set(true))
    cost = np.zeros((len(labels_p), len(labels_t)))
    for i, p in enumerate(labels_p):
        for j, t in enumerate(labels_t):
            cost[i, j] = -np.sum((pred == p) & (true == t))
    ri, ci = linear_sum_assignment(cost)
    mapping = {labels_p[r]: labels_t[c] for r, c in zip(ri, ci)}
    remapped = np.array([mapping.get(p, -1) for p in pred])
    return float(np.mean(remapped == true)), mapping


def build_parser():
    p = argparse.ArgumentParser(description="Score seg-ID embedding vs held-out 3D global_id.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--min-global-id-purity", type=float, default=0.0,
                   help="optional filter for weak geometry labels when global_id_purity is present")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    meta = pd.read_parquet(Path(args.dataset) / "crop_metadata.parquet")
    emb = embed(args.dataset, args.model, args.device)
    meta = meta.reset_index(drop=True)
    assert len(meta) == len(emb)

    purity = meta.get("global_id_purity", pd.Series(1.0, index=meta.index))
    m = meta[(meta.qc_status == "clean_train") & (meta.global_id >= 0)
             & (purity.astype(float) >= float(args.min_global_id_purity))].copy()
    X = emb[m.index.to_numpy()]
    y = m.global_id.to_numpy()
    n_groups = len(set(y))

    # crop-level
    sil = float(silhouette_score(X, y)) if n_groups >= 2 and len(X) > n_groups else float("nan")
    km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
    ari = float(adjusted_rand_score(y, km.labels_))
    crop_acc, _ = remap_accuracy(km.labels_, y)

    # tracklet-level (mean embedding per view_tracklet -> nearest of 3 cluster centroids)
    trk = m.groupby("view_tracklet")
    t_emb = np.stack([X[m.index.get_indexer(g.index)].mean(axis=0) for _, g in
                      m.reset_index().groupby("view_tracklet")])
    t_true = np.array([int(g.global_id.iloc[0]) for _, g in m.groupby("view_tracklet")])
    t_pred = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(t_emb)
    t_ari = float(adjusted_rand_score(t_true, t_pred))
    t_acc, _ = remap_accuracy(t_pred, t_true)

    # cannot-link satisfaction: same view+frame, different tracklet -> different cluster?
    crop_cluster = {int(i): int(c) for i, c in zip(m.index, km.labels_)}
    sat = tot = 0
    for (_v, _f), g in m.groupby(["view", "frame"]):
        idxs = list(g.index)
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                if m.loc[idxs[a], "view_tracklet"] != m.loc[idxs[b], "view_tracklet"]:
                    tot += 1
                    sat += int(crop_cluster[idxs[a]] != crop_cluster[idxs[b]])
    cannot_link = round(sat / tot, 4) if tot else float("nan")

    report = {
        "n_clean_crops": int(len(m)), "n_view_tracklets": int(m.view_tracklet.nunique()),
        "n_true_groups": n_groups,
        "silhouette_global_id": round(sil, 4),
        "crop_kmeans_ARI": round(ari, 4),
        "crop_accuracy_remapped": round(crop_acc, 4),
        "tracklet_kmeans_ARI": round(t_ari, 4),
        "tracklet_accuracy_remapped": round(t_acc, 4),
        "cannot_link_satisfied": cannot_link,
        "per_view_crop_counts": {k: int(v) for k, v in m.view.value_counts().items()},
        "min_global_id_purity": float(args.min_global_id_purity),
        "interpretation": ("ARI/silhouette are remap-free. silhouette>=0.91 ~ <1% per-frame "
                           "(idtracker); ~0.73 was moderate. Compare crop_accuracy_remapped to "
                           "the prior 0.72 and tracklet to 0.77."),
    }
    (out_dir / "segid_eval_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # 2D PCA scatter coloured by global_id
    try:
        from sklearn.decomposition import PCA
        P = PCA(n_components=2).fit_transform(X)
        fig, ax = plt.subplots(figsize=(6, 5))
        for gid in sorted(set(y)):
            mk = y == gid
            ax.scatter(P[mk, 0], P[mk, 1], s=4, alpha=0.4, label=f"fish {gid}")
        ax.legend(); ax.set_title(f"seg-ID embedding (PCA) by 3D global_id\n"
                                  f"silhouette={sil:.2f} ARI={ari:.2f} trk-acc(remap)={t_acc:.2f}")
        fig.tight_layout(); fig.savefig(out_dir / "segid_embedding_pca.png", dpi=120); plt.close(fig)
    except Exception as e:
        print("pca plot skipped:", e)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
