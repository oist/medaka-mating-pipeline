from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SmallMedakaEmbeddingNet(nn.Module):
    def __init__(self, embedding_dim: int = 64) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2, padding=2),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 48, 3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 96, 3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(96, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x).flatten(1)
        z = self.head(z)
        return F.normalize(z, dim=1)


def contrastive_loss(
    emb_a: torch.Tensor,
    emb_b: torch.Tensor,
    labels: torch.Tensor,
    *,
    margin: float = 1.0,
) -> torch.Tensor:
    d = F.pairwise_distance(emb_a, emb_b)
    pos = labels * d.pow(2)
    neg = (1.0 - labels) * F.relu(margin - d).pow(2)
    return (pos + neg).mean()

