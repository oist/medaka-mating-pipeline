"""Pure-PyTorch Precise RoI Pooling — a drop-in replacement for UDMT's CUDA
``PreciseRoIPooling`` op (ltr/external/PreciseRoIPooling).

Precise RoI Pooling (Jiang et al., "Acquisition of Localization Confidence for Accurate
Object Detection", ECCV 2018) pools each output bin as the *integral* of the bilinearly
interpolated feature map over the continuous bin region, divided by the bin area — unlike
RoIAlign which averages a fixed set of sample points. Critically it is differentiable w.r.t.
the RoI **coordinates**, which DiMP's IoU-Net relies on (it refines boxes by gradient ascent
on predicted IoU at inference).

The integral separates along x and y because bilinear interpolation is a product of 1-D
tent kernels ``Lambda(t) = max(0, 1 - |t|)``. For a bin spanning ``[a, b]`` along an axis,
the weight on grid point ``u`` is ``Phi(b - u) - Phi(a - u)`` where ``Phi`` is the
antiderivative of ``Lambda``. Pooling is then a pair of batched contractions
``Wy @ feat @ Wx^T`` divided by the bin area. Expressed entirely in differentiable torch
ops, so autograd supplies *both* the feature and coordinate gradients the CUDA kernel
hand-wrote — no custom build, runs anywhere torch runs.
"""
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["PrRoIPool2D", "prroi_pool2d"]


def _phi(t: torch.Tensor) -> torch.Tensor:
    """Antiderivative of the 1-D tent kernel ``Lambda(t) = max(0, 1 - |t|)``.

    ``Phi(t) = ∫_{-inf}^{t} Lambda``: flat 0 for ``t <= -1``, 1 for ``t >= 1``, with
    ``Phi(-1)=0``, ``Phi(0)=0.5``, ``Phi(1)=1``. Smooth (C1) and differentiable.
    """
    tc = t.clamp(-1.0, 1.0)
    return 0.5 + tc - 0.5 * tc.abs() * tc


def _axis_weights(lo: torch.Tensor, hi: torch.Tensor, size: int) -> torch.Tensor:
    """Integral of every grid-point tent over every bin along one axis.

    ``lo``, ``hi``: ``(N, P)`` continuous bin edges in feature coords; ``size``: grid length.
    Returns ``(N, P, size)`` where entry ``[n, p, u]`` is ``∫_{lo}^{hi} Lambda(x - u) dx``.
    """
    u = torch.arange(size, dtype=lo.dtype, device=lo.device)         # (size,)
    return _phi(hi.unsqueeze(-1) - u) - _phi(lo.unsqueeze(-1) - u)   # (N, P, size)


def prroi_pool2d(features: torch.Tensor, rois: torch.Tensor,
                 pooled_height: int, pooled_width: int, spatial_scale: float) -> torch.Tensor:
    """Precise RoI Pooling.

    Args:
        features: ``(B, C, H, W)`` feature map.
        rois: ``(N, 5)`` rows ``[batch_index, x1, y1, x2, y2]`` in *input* coordinates
            (scaled into feature coords by ``spatial_scale``).
        pooled_height, pooled_width: output bin grid.
        spatial_scale: feature_coord = input_coord * spatial_scale.

    Returns:
        ``(N, C, pooled_height, pooled_width)``. Differentiable w.r.t. ``features`` and ``rois``.
    """
    if features.dtype not in (torch.float32, torch.float64):
        features = features.float()
    B, C, H, W = features.shape
    N = rois.shape[0]
    ph, pw = int(pooled_height), int(pooled_width)
    s = float(spatial_scale)

    rois = rois.to(features.dtype)
    batch_idx = rois[:, 0].long()
    # scale to feature coords and clamp to the valid grid (matches the CUDA op's boundary)
    x1 = (rois[:, 1] * s).clamp(0, W - 1)
    y1 = (rois[:, 2] * s).clamp(0, H - 1)
    x2 = (rois[:, 3] * s).clamp(0, W - 1)
    y2 = (rois[:, 4] * s).clamp(0, H - 1)

    jx = torch.arange(pw + 1, dtype=features.dtype, device=features.device)
    iy = torch.arange(ph + 1, dtype=features.dtype, device=features.device)
    xe = x1.unsqueeze(1) + (x2 - x1).unsqueeze(1) * jx / pw          # (N, pw+1)
    ye = y1.unsqueeze(1) + (y2 - y1).unsqueeze(1) * iy / ph          # (N, ph+1)

    Wx = _axis_weights(xe[:, :-1], xe[:, 1:], W)                     # (N, pw, W)
    Wy = _axis_weights(ye[:, :-1], ye[:, 1:], H)                     # (N, ph, H)

    feat = features[batch_idx]                                       # (N, C, H, W)
    # integral over each bin: out[n,c,i,j] = sum_{v,u} Wy[n,i,v] feat[n,c,v,u] Wx[n,j,u]
    tmp = torch.einsum("niv,ncvu->nciu", Wy, feat)                   # (N, C, ph, W)
    out = torch.einsum("nciu,nju->ncij", tmp, Wx)                   # (N, C, ph, pw)

    bin_area = ((x2 - x1) / pw) * ((y2 - y1) / ph)                   # (N,)
    return out / bin_area.clamp_min(1e-6).view(N, 1, 1, 1)


class PrRoIPool2D(nn.Module):
    """Pure-PyTorch Precise RoI Pooling, drop-in for ``ltr.external.PreciseRoIPooling``.

    Differentiable w.r.t. both the input feature map and the RoI coordinates.
    """

    def __init__(self, pooled_height: int, pooled_width: int, spatial_scale: float):
        super().__init__()
        self.pooled_height = int(pooled_height)
        self.pooled_width = int(pooled_width)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
        return prroi_pool2d(features, rois, self.pooled_height, self.pooled_width, self.spatial_scale)

    def extra_repr(self) -> str:
        return "kernel_size=({}, {}), spatial_scale={}".format(
            self.pooled_height, self.pooled_width, self.spatial_scale)
