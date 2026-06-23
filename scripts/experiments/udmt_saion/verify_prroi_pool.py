"""Verify the pure-PyTorch PrRoIPool (UDMT.stnet.prroi_pool) against UDMT's CUDA op.

Run on a GPU box that has the original PreciseRoIPooling build (e.g. saion)::

    python verify_prroi_pool.py --udmt-root /bucket/.../UDMT --prroi-module /path/to/prroi_pool.py

Checks (1) forward parity, (2) d/d-features parity, (3) d/d-rois parity vs the CUDA kernel,
and (4) double-precision gradcheck of the pure implementation.
"""
import argparse
import importlib.util
import sys

import torch


def _load_pure(path):
    spec = importlib.util.spec_from_file_location("pure_prroi", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.prroi_pool2d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--udmt-root", required=True)
    p.add_argument("--prroi-module", required=True, help="path to the pure prroi_pool.py")
    args = p.parse_args()

    pure = _load_pure(args.prroi_module)
    # ltr/__init__ imports the `udmt` package AND bare `ltr.`/`pytracking.` — need both roots
    sys.path[:0] = [args.udmt_root, f"{args.udmt_root}/udmt/gui/tabs/ST_Net"]
    from ltr.external.PreciseRoIPooling.pytorch.prroi_pool.functional import prroi_pool2d as cuda_pool

    torch.set_grad_enabled(True)   # UDMT's interactive_demo import disables grad globally
    torch.manual_seed(0)
    dev = "cuda"
    ph, pw, scale = 3, 3, 0.5
    for trial in range(3):
        B, C, H, W, N = 2, 5, 16, 16, 6
        feat = torch.randn(B, C, H, W, device=dev)
        bidx = torch.randint(0, B, (N, 1), device=dev).float()
        xy = torch.rand(N, 2, device=dev) * 10
        wh = torch.rand(N, 2, device=dev) * 12 + 4
        rois = torch.cat([bidx, xy, xy + wh], dim=1)

        # forward parity
        fc = feat.clone().requires_grad_(True)
        rc = rois.clone().requires_grad_(True)
        fp = feat.clone().requires_grad_(True)
        rp = rois.clone().requires_grad_(True)
        out_c = cuda_pool(fc, rc, ph, pw, scale)
        out_p = pure(fp, rp, ph, pw, scale)
        f_err = (out_c - out_p).abs().max().item()

        # backward parity (features + rois)
        g = torch.randn_like(out_c)
        out_c.backward(g)
        out_p.backward(g)
        gf_err = (fc.grad - fp.grad).abs().max().item()
        gr_err = (rc.grad[:, 1:] - rp.grad[:, 1:]).abs().max().item()
        print(f"trial {trial}: fwd_max_err={f_err:.3e}  dfeat_max_err={gf_err:.3e}  "
              f"droi_max_err={gr_err:.3e}")

    # gradcheck (double precision, CPU)
    feat = torch.randn(2, 2, 8, 8, dtype=torch.double, requires_grad=True)
    coords = torch.tensor([[1.0, 1.0, 6.0, 6.0], [0.5, 0.5, 7.0, 5.0]],
                          dtype=torch.double, requires_grad=True)
    bidx = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    ok = torch.autograd.gradcheck(
        lambda f, c: pure(f, torch.cat([bidx, c], 1), 3, 3, 1.0), (feat, coords),
        eps=1e-6, atol=1e-4)
    print(f"gradcheck (feat + coords): {'PASS' if ok else 'FAIL'}")
    print("PRROI_VERIFY_DONE")


if __name__ == "__main__":
    main()
