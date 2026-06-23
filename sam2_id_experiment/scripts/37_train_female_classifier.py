from __future__ import annotations

"""Train the production FEMALE classifier (female vs male) on seg-ID crops.

ResNet18 (ImageNet-pretrained) + 2-logit head, class 1 = female. Trained on the
reliable female/male split (female = the cleanly-separable slot; males pooled).
Tracklet-grouped train/val split so the reported accuracy is honest (no crop leak).
The two males are deliberately NOT distinguished -- only the validated capability.

Run::

    python -m sam2_id_experiment.scripts.37_train_female_classifier \
        --dataset <pooled_sides_full> --female-gid 1 --out-dir <model_female> --epochs 20
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from sam2_id_experiment.src.id_model.resnet_embedder import EmbeddingClassifier, ResNet18Embedder


class CropLabelDataset(Dataset):
    def __init__(self, crops, labels, augment=True):
        self.crops, self.labels, self.augment = crops, labels, augment

    def __len__(self):
        return len(self.labels)

    def _aug(self, im):
        if not self.augment:
            return im
        if np.random.rand() < 0.5:
            im = im[:, ::-1]
        if np.random.rand() < 0.5:
            im = np.clip(im * np.float32(np.random.uniform(0.85, 1.15))
                         + np.float32(np.random.uniform(-0.05, 0.05)), 0, 1)
        return np.ascontiguousarray(im)

    def __getitem__(self, i):
        im = self._aug(self.crops[i].astype(np.float32) / 255.0)
        return torch.from_numpy(im).permute(2, 0, 1), torch.tensor(int(self.labels[i]))


def _balacc(y, p):
    y, p = np.asarray(y), np.asarray(p)
    accs = [np.mean(p[y == c] == c) for c in np.unique(y) if (y == c).any()]
    return float(np.mean(accs))


def build_parser():
    p = argparse.ArgumentParser(description="Train the female-vs-male anchor classifier.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--female-gid", type=int, default=1)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1.5e-4)
    p.add_argument("--val-frac", type=float, default=0.25)
    p.add_argument("--device", default="cuda")
    p.add_argument("--min-global-id-purity", type=float, default=0.9,
                   help="drop weak labels whose modal 3D global_id purity is below this threshold")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    with h5py.File(Path(args.dataset) / "crops.h5", "r") as h:
        crops = h["crops"][...]
    meta = pd.read_parquet(Path(args.dataset) / "crop_metadata.parquet").reset_index(drop=True)
    purity = meta.get("global_id_purity", pd.Series(1.0, index=meta.index))
    use = meta[(meta.qc_status == "clean_train") & (meta.global_id >= 0)
               & (purity.astype(float) >= float(args.min_global_id_purity))].copy()
    use["label"] = (use.global_id == args.female_gid).astype(int)

    # tracklet-grouped train/val split
    rng = np.random.default_rng(0)
    tracks = use.view_tracklet.unique()
    val_tracks = set(rng.choice(tracks, size=max(1, int(len(tracks) * args.val_frac)), replace=False))
    tr = use[~use.view_tracklet.isin(val_tracks)]
    va = use[use.view_tracklet.isin(val_tracks)]
    print(f"train {len(tr)} crops / {tr.view_tracklet.nunique()} trk; "
          f"val {len(va)} crops / {va.view_tracklet.nunique()} trk", flush=True)

    dev = args.device
    model = EmbeddingClassifier(ResNet18Embedder(embedding_dim=128, pretrained=True), 2).to(dev)
    # class weights (female is ~1/3)
    w = torch.tensor([1.0, float((tr.label == 0).sum()) / max(int((tr.label == 1).sum()), 1)],
                     dtype=torch.float32).to(dev)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    dl = DataLoader(CropLabelDataset(crops[tr.index.to_numpy()], tr.label.to_numpy(), True),
                    batch_size=args.batch_size, shuffle=True, drop_last=True)
    for ep in range(args.epochs):
        model.train(); tot = n = 0
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
            tot += loss.item() * len(y); n += len(y)
        sched.step()
        print(f"  epoch {ep}: loss {tot/max(n,1):.4f}", flush=True)

    # validate (held-out tracklets), per-crop and per-tracklet
    model.eval()
    vx = crops[va.index.to_numpy()].astype(np.float32) / 255.0
    preds, probs = [], []
    with torch.no_grad():
        for s in range(0, len(vx), 256):
            b = torch.from_numpy(vx[s:s + 256]).permute(0, 3, 1, 2).to(dev)
            pr = torch.softmax(model(b), 1)[:, 1].cpu().numpy()
            probs.append(pr); preds.append((pr >= 0.5).astype(int))
    va = va.copy(); va["prob"] = np.concatenate(probs); va["pred"] = np.concatenate(preds)
    crop_bal = _balacc(va.label.to_numpy(), va.pred.to_numpy())
    trk = va.groupby("view_tracklet").agg(label=("label", "first"), prob=("prob", "mean"))
    trk["pred"] = (trk.prob >= 0.5).astype(int)
    trk_bal = _balacc(trk.label.to_numpy(), trk.pred.to_numpy())

    torch.save({"model_state": model.state_dict(), "embedding_dim": 128, "arch": "resnet18",
                "female_label": 1, "female_gid_trained": args.female_gid}, out / "model.pt")
    report = {"val_crop_balacc": round(crop_bal, 3), "val_tracklet_balacc": round(trk_bal, 3),
              "n_train_crops": int(len(tr)), "n_val_crops": int(len(va)),
              "n_val_tracklets": int(len(trk)),
              "min_global_id_purity": float(args.min_global_id_purity)}
    (out / "female_classifier_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
