"""ResNet18 backbone producing an L2-normalized 8-D embedding."""
from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18


class IdentityEmbedder(nn.Module):
    """
    1-channel ResNet18 whose `fc` has been replaced with a `Linear(512, d)`
    head; outputs are L2-normalized so cosine and Euclidean distance agree.
    """

    def __init__(self, embedding_dim: int = 8):
        super().__init__()
        backbone = resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        backbone.fc = nn.Linear(backbone.fc.in_features, embedding_dim)
        self.backbone = backbone
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        return nn.functional.normalize(z, dim=1)
