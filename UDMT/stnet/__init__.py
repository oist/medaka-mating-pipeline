"""Portable ST-Net / TrDiMP support code ported from UDMT's PyTracking/LTR stack.

The original ST-Net lives in UDMT's `udmt/gui/tabs/ST_Net` (254 files, ~43k LOC of the
PyTracking + LTR framework) and is pinned to Linux / py3.8 / torch 2.1.1 by one custom CUDA
op — Precise RoI Pooling. This package re-implements that op in pure PyTorch (`prroi_pool`)
so patched ST-Net clones can build and run on modern torch (incl. Blackwell) without the
custom CUDA extension. Vendoring the minimal TrDiMP inference path is still future work.
"""

__all__ = ["PrRoIPool2D", "prroi_pool2d"]


def __getattr__(name: str):
    if name in __all__:
        from .prroi_pool import PrRoIPool2D, prroi_pool2d

        return {"PrRoIPool2D": PrRoIPool2D, "prroi_pool2d": prroi_pool2d}[name]
    raise AttributeError(name)
