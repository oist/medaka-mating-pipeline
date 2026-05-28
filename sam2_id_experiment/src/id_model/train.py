from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from sam2_id_experiment.src.id_model.model import SmallMedakaEmbeddingNet, contrastive_loss


class CropPairDataset(Dataset):
    def __init__(self, dataset_dir: str | Path) -> None:
        import h5py

        self.dataset_dir = Path(dataset_dir)
        with h5py.File(self.dataset_dir / "crops.h5", "r") as h5:
            self.crops = np.asarray(h5["crops"][...], dtype=np.float32) / 255.0
        self.pairs = pd.read_parquet(self.dataset_dir / "pairs.parquet")
        if self.pairs.empty:
            raise RuntimeError("pairs.parquet is empty; cannot train ID model.")

    def __len__(self) -> int:
        return int(self.pairs.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.pairs.iloc[idx]
        a = self.crops[int(row["crop_a_id"])]
        b = self.crops[int(row["crop_b_id"])]
        a_t = torch.from_numpy(a).permute(2, 0, 1)
        b_t = torch.from_numpy(b).permute(2, 0, 1)
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return a_t, b_t, label


def train_embedding_model(
    *,
    dataset_dir: str | Path,
    output_dir: str | Path,
    embedding_dim: int = 64,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
) -> dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = CropPairDataset(dataset_dir)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    model = SmallMedakaEmbeddingNet(embedding_dim=embedding_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    log_path = out_dir / "training_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["epoch", "step", "loss"])
        writer.writeheader()
        global_step = 0
        for epoch in range(1, epochs + 1):
            model.train()
            for a, b, y in loader:
                a = a.to(device)
                b = b.to(device)
                y = y.to(device)
                opt.zero_grad(set_to_none=True)
                loss = contrastive_loss(model(a), model(b), y)
                loss.backward()
                opt.step()
                writer.writerow({"epoch": epoch, "step": global_step, "loss": float(loss.detach().cpu())})
                global_step += 1

    model_path = out_dir / "model.pt"
    torch.save({
        "model_state": model.state_dict(),
        "embedding_dim": embedding_dim,
    }, model_path)
    preview_path = out_dir / "embedding_preview.png"
    _write_embedding_preview(model, ds, preview_path, device=device)
    return {
        "model_path": str(model_path),
        "training_log": str(log_path),
        "embedding_preview": str(preview_path),
        "n_pairs": len(ds),
    }


def _write_embedding_preview(
    model: SmallMedakaEmbeddingNet,
    ds: CropPairDataset,
    out_path: Path,
    *,
    device: str,
) -> None:
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    meta_path = ds.dataset_dir / "crop_metadata.parquet"
    if not meta_path.exists():
        return
    meta = pd.read_parquet(meta_path)
    if meta.empty:
        return

    max_points = min(2000, len(ds.crops))
    idx = np.linspace(0, len(ds.crops) - 1, max_points).astype(int)
    crops = torch.from_numpy(ds.crops[idx]).permute(0, 3, 1, 2).to(device)
    model.eval()
    with torch.no_grad():
        emb = model(crops).cpu().numpy()
    xy = PCA(n_components=2).fit_transform(emb) if emb.shape[0] > 2 else np.pad(emb[:, :1], ((0, 0), (0, 1)))
    labels = meta.iloc[idx]["tracklet_id"].astype(str).to_numpy()

    fig, ax = plt.subplots(figsize=(6, 5))
    for label in sorted(set(labels)):
        mask = labels == label
        ax.scatter(xy[mask, 0], xy[mask, 1], s=10, label=f"tracklet {label}", alpha=0.75)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("ID crop embedding preview")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
