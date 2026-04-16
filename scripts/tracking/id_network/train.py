"""Self-supervised contrastive training of the identity embedder."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .model import IdentityEmbedder


def _augment(crop: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Cheap augmentation: random rotation (any angle, since the top view has
    no canonical orientation) plus small contrast jitter. Input/output are
    `(size, size)` uint8."""
    size = crop.shape[-1]
    angle = float(rng.uniform(0, 360))
    c, s = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
    # 2x3 affine around the crop center.
    import cv2

    M = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1.0)
    rotated = cv2.warpAffine(crop, M, (size, size), borderValue=0)
    gain = float(rng.uniform(0.8, 1.2))
    bias = float(rng.uniform(-10, 10))
    out = np.clip(rotated.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
    return out


class _PairDataset(Dataset):
    """Serves `(crop_a, crop_b, label)` triples where label in {0: diff, 1: same}."""

    def __init__(
        self,
        images: np.ndarray,
        same_pairs: list[tuple[int, int]],
        diff_pairs: list[tuple[int, int]],
        augment: bool = True,
        rng_seed: int = 0,
    ):
        self.images = images
        self.pairs = [(i, j, 1) for i, j in same_pairs] + [
            (i, j, 0) for i, j in diff_pairs
        ]
        self.augment = augment
        self._rng = np.random.default_rng(rng_seed)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        i, j, label = self.pairs[idx]
        a = self.images[i]
        b = self.images[j]
        if self.augment:
            a = _augment(a, self._rng)
            b = _augment(b, self._rng)
        a_t = torch.from_numpy(a.astype(np.float32) / 255.0).unsqueeze(0)
        b_t = torch.from_numpy(b.astype(np.float32) / 255.0).unsqueeze(0)
        return a_t, b_t, torch.tensor(label, dtype=torch.float32)


@dataclass
class TrainConfig:
    epochs: int = 5
    batch_size: int = 64
    lr: float = 1e-3
    embedding_dim: int = 8
    device: str = "cuda"
    num_workers: int = 0
    margin: float = 0.5  # margin for the diff-pair hinge in contrastive loss


def _contrastive_loss(
    za: torch.Tensor, zb: torch.Tensor, label: torch.Tensor, margin: float
) -> torch.Tensor:
    """
    Standard contrastive loss on L2-normalized embeddings:
        same:  0.5 * ||za - zb||^2
        diff:  0.5 * max(0, margin - ||za - zb||)^2

    `za`, `zb`: (B, d), unit-norm.  `label`: (B,), 1.0 for same, 0.0 for diff.
    """
    d = (za - zb).norm(dim=1)
    loss_same = 0.5 * d.pow(2)
    loss_diff = 0.5 * torch.clamp(margin - d, min=0).pow(2)
    return (label * loss_same + (1.0 - label) * loss_diff).mean()


def train_embedder(
    images: np.ndarray,
    same_pairs: list[tuple[int, int]],
    diff_pairs: list[tuple[int, int]],
    config: TrainConfig | None = None,
) -> tuple[IdentityEmbedder, dict]:
    """
    Train `IdentityEmbedder` on mined pairs and return `(model, history)`.

    `images` is `(N, size, size)` uint8. The pair indices must reference rows
    of `images`.
    """
    config = config or TrainConfig()
    if not same_pairs or not diff_pairs:
        raise ValueError(
            f"Need both same and diff pairs to train "
            f"(got {len(same_pairs)} same, {len(diff_pairs)} diff)."
        )

    device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")
    model = IdentityEmbedder(embedding_dim=config.embedding_dim).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=config.lr)

    dataset = _PairDataset(images, same_pairs, diff_pairs, augment=True)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True,
    )

    history: dict = {"epoch_loss": [], "pair_accuracy": []}
    model.train()
    for epoch in range(config.epochs):
        epoch_loss = 0.0
        n_correct = 0
        n_total = 0
        for a, b, label in loader:
            a = a.to(device)
            b = b.to(device)
            label = label.to(device)
            za = model(a)
            zb = model(b)
            loss = _contrastive_loss(za, zb, label, margin=config.margin)
            optim.zero_grad()
            loss.backward()
            optim.step()

            with torch.no_grad():
                d = (za - zb).norm(dim=1)
                pred = (d < config.margin * 0.5).float()
                n_correct += int((pred == label).sum().item())
                n_total += int(label.numel())
                epoch_loss += float(loss.item()) * label.numel()

        avg_loss = epoch_loss / max(1, n_total)
        acc = n_correct / max(1, n_total)
        history["epoch_loss"].append(avg_loss)
        history["pair_accuracy"].append(acc)
        print(f"  epoch {epoch + 1}/{config.epochs}  loss={avg_loss:.4f}  pair_acc={acc:.3f}")

    model.eval()
    return model, history


def embed_all(
    model: IdentityEmbedder,
    images: np.ndarray,
    batch_size: int = 128,
    device: str | None = None,
) -> np.ndarray:
    """Run the embedder over `images` in mini-batches and return (N, d) floats."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(dev).eval()
    outs = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            x = torch.from_numpy(batch.astype(np.float32) / 255.0).unsqueeze(1).to(dev)
            outs.append(model(x).cpu().numpy())
    if not outs:
        return np.zeros((0, model.embedding_dim), dtype=np.float32)
    return np.concatenate(outs, axis=0)
