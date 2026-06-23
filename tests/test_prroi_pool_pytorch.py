"""Tests for the pure-PyTorch Precise RoI Pooling (UDMT.stnet.prroi_pool).

Numerical parity with the CUDA op is checked separately on a GPU box
(scripts/experiments/udmt_saion/verify_prroi_pool.py); here we check the math:
correct shapes, the exact-average property, and autograd gradcheck w.r.t. BOTH the
feature map and the RoI coordinates (the property DiMP's IoU-Net depends on).
"""
import torch

from UDMT.stnet.prroi_pool import PrRoIPool2D, prroi_pool2d


def test_shape_and_module():
    feat = torch.randn(2, 4, 9, 7)
    rois = torch.tensor([[0.0, 1.0, 1.0, 6.0, 5.0], [1.0, 0.0, 0.0, 6.0, 8.0]])
    out = PrRoIPool2D(3, 5, 1.0)(feat, rois)
    assert out.shape == (2, 4, 3, 5)


def test_constant_feature_averages_to_constant():
    # PrPool of a constant feature over any box equals that constant.
    feat = torch.full((1, 2, 10, 10), 0.37)
    rois = torch.tensor([[0.0, 2.3, 1.1, 8.7, 9.2]])
    out = prroi_pool2d(feat, rois, 3, 3, 1.0)
    assert torch.allclose(out, torch.full_like(out, 0.37), atol=1e-5)


def test_spatial_scale():
    # input-coord box halved by spatial_scale 0.5 lands on the same feature region.
    feat = torch.randn(1, 3, 8, 8)
    a = prroi_pool2d(feat, torch.tensor([[0.0, 2.0, 2.0, 6.0, 6.0]]), 3, 3, 1.0)
    b = prroi_pool2d(feat, torch.tensor([[0.0, 4.0, 4.0, 12.0, 12.0]]), 3, 3, 0.5)
    assert torch.allclose(a, b, atol=1e-5)


def test_gradcheck_features_and_coords():
    # Numerical parity vs the CUDA kernel (~1e-6 on fwd, d-feat AND d-roi) is checked on a
    # GPU box by scripts/experiments/udmt_saion/verify_prroi_pool.py. Here we gradcheck the
    # pure impl with off-kink coords (the tent kernel is only C1 at integer offsets, where
    # finite differences straddle a kink and falsely fail).
    torch.manual_seed(0)
    feat = torch.randn(2, 2, 8, 8, dtype=torch.double, requires_grad=True)
    coords = torch.tensor([[0.55, 0.55, 6.45, 6.45], [0.7, 0.3, 6.1, 5.7]],
                          dtype=torch.double, requires_grad=True)
    bidx = torch.tensor([[0.0], [1.0]], dtype=torch.double)

    def fn(f, c):
        return prroi_pool2d(f, torch.cat([bidx, c], dim=1), 3, 3, 1.0)

    assert torch.autograd.gradcheck(fn, (feat, coords), eps=1e-6, atol=1e-4)
