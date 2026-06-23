from __future__ import annotations

"""ImageNet-pretrained ResNet18 embedding backbone for medaka identity.

Much higher capacity than ``SmallMedakaEmbeddingNet`` (11.7M vs ~0.06M params) and,
critically, ImageNet-pretrained features generalise from far less data -- the right
backbone for the subtle 2-male appearance problem (idtracker.ai-class capacity).

Preprocessing is baked into ``forward`` so the existing crop pipeline (BGR uint8 ->
/255 float CHW) can be reused unchanged: it flips BGR->RGB and applies ImageNet
normalisation. Output is L2-normalised.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18Embedder(nn.Module):
    def __init__(self, embedding_dim: int = 128, pretrained: bool = True,
                 bgr_input: bool = True, freeze_until: str | None = None) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnet18(weights=weights)
        net.fc = nn.Linear(512, embedding_dim)
        self.net = net
        self.bgr_input = bgr_input
        self.embedding_dim = embedding_dim
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        if freeze_until:  # optionally freeze early blocks for tiny datasets
            freeze = True
            for name, p in self.net.named_parameters():
                if name.startswith(freeze_until):
                    freeze = False
                p.requires_grad = not freeze

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bgr_input:
            x = x.flip(1)
        x = (x - self.mean) / self.std
        return F.normalize(self.net(x), dim=1)


class EmbeddingClassifier(nn.Module):
    """Frozen/active embedder + linear softmax head (idtracker-style supervised head
    for accumulation: train on confidently constraint-propagated labels)."""

    def __init__(self, embedder: nn.Module, n_classes: int) -> None:
        super().__init__()
        self.embedder = embedder
        self.head = nn.Linear(embedder.embedding_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embedder(x))
