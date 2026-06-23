"""Verify DiMP/ATOM IoU-Net parity with pure-PyTorch PrRoIPool swapped in.

This checks the IoU-Net module on synthetic features/proposals, not a checkpointed
full-tracker run.

Run on a GPU box with UDMT's ST_Net build (saion)::

    python verify_iou_net.py --udmt-root /bucket/.../UDMT --prroi-module /path/to/prroi_pool.py
"""
import argparse
import importlib.util
import sys

import torch


def _load_pure(path):
    spec = importlib.util.spec_from_file_location("pure_prroi", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PrRoIPool2D


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--udmt-root", required=True)
    p.add_argument("--prroi-module", required=True)
    args = p.parse_args()

    PurePool = _load_pure(args.prroi_module)
    sys.path[:0] = [args.udmt_root, f"{args.udmt_root}/udmt/gui/tabs/ST_Net"]
    torch.set_grad_enabled(False)  # inference parity
    from ltr.models.bbreg.atom_iou_net import AtomIoUNet

    torch.manual_seed(0)
    net = AtomIoUNet().cuda().eval()

    # dummy ResNet block3/block4 features (stride 8 / 16) + boxes in image coords (x,y,w,h)
    c3 = torch.randn(1, 128, 36, 36, device="cuda")
    c4 = torch.randn(1, 256, 18, 18, device="cuda")
    bb1 = torch.tensor([[[80.0, 80.0, 60.0, 60.0]]], device="cuda")           # (img, seq, 4)
    proposals = torch.tensor([[[[80.0, 80.0, 60.0, 60.0],
                                [70.0, 90.0, 50.0, 55.0],
                                [100.0, 100.0, 40.0, 40.0]]]], device="cuda")  # (img, seq, K, 4)

    out_cuda = net([c3, c4], [c3, c4], bb1, proposals).clone()

    # swap every Precise-RoI-Pool layer for the pure implementation (same params)
    for name in ["prroi_pool3r", "prroi_pool3t", "prroi_pool4r", "prroi_pool4t"]:
        old = getattr(net, name)
        setattr(net, name, PurePool(old.pooled_height, old.pooled_width, old.spatial_scale).cuda())
    out_pure = net([c3, c4], [c3, c4], bb1, proposals).clone()

    err = (out_cuda - out_pure).abs().max().item()
    rel = err / (out_cuda.abs().max().item() + 1e-9)
    print(f"IoU-Net predicted-IoU: CUDA {out_cuda.flatten().tolist()}")
    print(f"IoU-Net predicted-IoU: pure {out_pure.flatten().tolist()}")
    print(f"IoU-Net pure-vs-CUDA  max_abs_err={err:.3e}  rel_err={rel:.3e}")
    print("IOU_NET_VERIFY_DONE")


if __name__ == "__main__":
    main()
