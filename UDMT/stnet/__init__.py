"""Refactored, portable ST-Net / TrDiMP tracker (ported from UDMT's PyTracking/LTR stack).

The original ST-Net lives in UDMT's `udmt/gui/tabs/ST_Net` (254 files, ~43k LOC of the
PyTracking + LTR framework) and is pinned to Linux / py3.8 / torch 2.1.1 by one custom CUDA
op — Precise RoI Pooling. This package re-implements that op in pure PyTorch (`prroi_pool`)
so the tracker builds and runs on modern torch (incl. Blackwell), then ports the minimal
TrDiMP inference path on top of it.
"""

from .prroi_pool import PrRoIPool2D, prroi_pool2d

__all__ = ["PrRoIPool2D", "prroi_pool2d"]
