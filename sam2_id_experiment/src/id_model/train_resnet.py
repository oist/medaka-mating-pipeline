from __future__ import annotations

"""Contrastive trainer for the ResNet18 embedder (reuses the crops/pairs format).

Same self-supervised pairs as the small model (must-link / cannot-link), but a
higher-capacity ImageNet-pretrained backbone and a lower LR suited to fine-tuning.
Light on-the-fly augmentation (flip / small rotation) improves generalisation from
limited tracklets.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sam2_id_experiment.src.id_model.resnet_embedder import ResNet18Embedder


class CropPairDataset(Dataset):
    def __init__(self, dataset_dir, augment: bool = True):
        import h5py
        self.dir = Path(dataset_dir)
        with h5py.File(self.dir / "crops.h5", "r") as h5:
            self.crops = np.asarray(h5["crops"][...], dtype=np.float32) / 255.0
        self.pairs = pd.read_parquet(self.dir / "pairs.parquet")
        if self.pairs.empty:
            raise RuntimeError("pairs.parquet is empty")
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def _aug(self, im):  # im: (H,W,3)
        # Conservative: horizontal flip + small brightness/contrast jitter ONLY.
        # NO rotation / vertical flip -- medaka have a strong head-tail orientation
        # and the female's distinguishing cue (fuller belly) is orientation-dependent,
        # so rotating crops destroys the identity signal.
        if not self.augment:
            return im
        if np.random.rand() < 0.5:
            im = im[:, ::-1]
        if np.random.rand() < 0.5:
            g = np.float32(np.random.uniform(0.85, 1.15))
            b = np.float32(np.random.uniform(-0.05, 0.05))
            im = np.clip(im * g + b, 0.0, 1.0)
        return np.ascontiguousarray(im)

    def __getitem__(self, idx):
        r = self.pairs.iloc[idx]
        a = self._aug(self.crops[int(r["crop_a_id"])])
        b = self._aug(self.crops[int(r["crop_b_id"])])
        return (torch.from_numpy(a).permute(2, 0, 1),
                torch.from_numpy(b).permute(2, 0, 1),
                torch.tensor(float(r["label"])))


def contrastive_loss(za, zb, label, margin=1.0):
    d = F.pairwise_distance(za, zb)
    pos = label * d.pow(2)
    neg = (1.0 - label) * F.relu(margin - d).pow(2)
    return (pos + neg).mean()


def train_resnet_embedding(dataset_dir, output_dir, *, epochs=12, batch_size=128,
                           lr=1e-4, embedding_dim=128, device="cuda", pretrained=True,
                           augment=True, num_workers=0):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    ds = CropPairDataset(dataset_dir, augment=augment)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    model = ResNet18Embedder(embedding_dim=embedding_dim, pretrained=pretrained).to(device)
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    log = []
    model.train()
    for ep in range(epochs):
        tot = n = 0
        for a, b, y in dl:
            a, b, y = a.to(device), b.to(device), y.to(device)
            opt.zero_grad()
            loss = contrastive_loss(model(a), model(b), y)
            loss.backward(); opt.step()
            tot += loss.item() * len(y); n += len(y)
        sched.step()
        log.append({"epoch": ep, "loss": tot / max(n, 1)})
        print(f"  epoch {ep}: loss {tot/max(n,1):.4f}", flush=True)
    torch.save({"model_state": model.state_dict(), "embedding_dim": embedding_dim,
                "arch": "resnet18", "pretrained": pretrained}, out / "model.pt")
    pd.DataFrame(log).to_csv(out / "training_log.csv", index=False)
    (out / "train_summary.json").write_text(json.dumps(
        {"n_pairs": len(ds), "epochs": epochs, "lr": lr, "embedding_dim": embedding_dim,
         "final_loss": log[-1]["loss"]}, indent=2), encoding="utf-8")
    return {"model_path": str(out / "model.pt"), "n_pairs": len(ds), "final_loss": log[-1]["loss"]}
