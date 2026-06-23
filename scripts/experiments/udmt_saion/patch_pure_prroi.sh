#!/usr/bin/env bash
# Swap UDMT's CUDA PreciseRoIPooling for the pure-PyTorch drop-in (UDMT/stnet/prroi_pool.py),
# so the whole ST-Net / TrDiMP stack builds and runs on modern torch with NO nvcc / custom
# CUDA op (incl. Blackwell, and CPU). Every consumer (atom_iou_net, target_classifier
# initializer) imports PrRoIPool2D -> functional.prroi_pool2d, so replacing that one function
# is the entire swap. Idempotent; original CUDA version kept as functional.py.cuda-orig.
#
#   patch_pure_prroi.sh <UDMT_ROOT> <path/to/prroi_pool.py>
set -e
U=${1:-/bucket/StephensU/medaka/udmt/UDMT}
PURE=${2:?usage: patch_pure_prroi.sh <UDMT_ROOT> <path/to/UDMT/stnet/prroi_pool.py>}
D=$U/udmt/gui/tabs/ST_Net/ltr/external/PreciseRoIPooling/pytorch/prroi_pool
cp "$PURE" "$D/pure_prroi.py"
[ -f "$D/functional.py.cuda-orig" ] || cp "$D/functional.py" "$D/functional.py.cuda-orig"
cat > "$D/functional.py" <<'PY'
# Patched: use the pure-PyTorch Precise RoI Pooling (UDMT/stnet/prroi_pool.py) instead of the
# CUDA kernel -- numerically matched to the original to ~1e-6 (fwd + d-feat + d-roi) and the
# TrDiMP IoU-Net to ~1e-8. Original CUDA version preserved as functional.py.cuda-orig.
from .pure_prroi import prroi_pool2d
__all__ = ['prroi_pool2d']
PY
echo "swapped PreciseRoIPooling -> pure PyTorch (orig: $D/functional.py.cuda-orig)"
