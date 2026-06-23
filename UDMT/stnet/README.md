# `UDMT/stnet` — portable ST-Net / TrDiMP refactor

UDMT's actual tracking core is **ST-Net / TrDiMP** (a TransformerMeetsTracker DiMP: ResNet-50
backbone + target classifier + IoU-Net + transformer, self-trained per video). It lives in the
upstream clone at `udmt/gui/tabs/ST_Net` — **254 files, ~43k LOC** of the PyTracking + LTR
framework — and was pinned to **Linux / py3.8 / torch 2.1.1** by two things:

1. one **custom CUDA op** — *Precise RoI Pooling* — which won't build on modern toolchains
   (e.g. Blackwell / RTX 50-series), and
2. deep **GUI coupling** — the tracker imports itself through the `udmt.gui.*` package, dragging
   in Qt (qtpy / PySide6) on any import.

This package refactors both away so the stack is importable and runnable as **plain, portable
PyTorch** (modern torch, CPU or any GPU, no custom build, no GUI).

## What's here

| File | Role | Status |
|---|---|---|
| `prroi_pool.py` | **Pure-PyTorch Precise RoI Pooling** — drop-in for the CUDA op | ✅ validated |
| `__init__.py` | exports `PrRoIPool2D`, `prroi_pool2d` | |
| `../../tests/test_prroi_pool_pytorch.py` | shape / average / spatial-scale / **gradcheck** | |
| `../../scripts/experiments/udmt_saion/verify_prroi_pool.py` | **CUDA-parity** harness (GPU) | ✅ 1e-6 |
| `../../scripts/experiments/udmt_saion/verify_iou_net.py` | full **IoU-Net** parity (GPU) | ✅ 1e-8 |
| `../../scripts/experiments/udmt_saion/patch_pure_prroi.sh` | swap clone's CUDA op → pure | |
| `../../scripts/experiments/udmt_saion/decouple_stnet_imports.sh` | strip the GUI/`..`-relative imports | |

## The keystone: pure Precise RoI Pooling

PrRoIPool pools each output bin as the *integral* of the bilinearly-interpolated feature over the
continuous bin region (not fixed sample points like RoIAlign), and is differentiable w.r.t. the
RoI **coordinates** — which the IoU-Net relies on (it refines boxes by gradient ascent on predicted
IoU at inference). That integral has a closed form (bilinear = a product of 1-D tent kernels whose
antiderivative is analytic), so `prroi_pool.py` computes it in pure differentiable torch and lets
**autograd** supply both the feature and coordinate gradients the CUDA kernel hand-wrote.

**Validated against the original CUDA kernel on a V100:**

| check | result |
|---|---|
| forward | max abs err **≈ 7e-7** |
| d / d-features | **≈ 6e-7** |
| d / d-roi-coords | **≈ 1e-6** |
| full TrDiMP IoU-Net predicted-IoU | **≈ 1.3e-8** |
| gradcheck (double, CPU) | PASS |

i.e. numerically identical at float32 epsilon — with no nvcc, no custom build, runnable on CPU.

## Making a UDMT clone portable

Two idempotent scripts transform an upstream `ST_Net` tree into the portable form:

```bash
# 1) replace the CUDA PreciseRoIPooling with the pure op everywhere
bash scripts/experiments/udmt_saion/patch_pure_prroi.sh  <UDMT_ROOT>  UDMT/stnet/prroi_pool.py
# 2) rewrite the GUI/`..`-relative imports to bare ltr.* / pytracking.*
bash scripts/experiments/udmt_saion/decouple_stnet_imports.sh  <UDMT_ROOT>/udmt/gui/tabs/ST_Net
```

After both, with `ST_Net` on `sys.path`, the whole stack imports **standalone, GUI-free, on CPU**:

```python
import sys, types
# the tracker still imports PySide6 only for an unused (vis_flag=False) visualizer -> stub it
for m in ["PySide6", "PySide6.QtGui", "PySide6.QtWidgets", "PySide6.QtCore"]:
    sys.modules[m] = types.ModuleType(m)
sys.path.insert(0, "<ST_Net>")
from pytracking.run_tracker import run_tracking          # tracker entry
from pytracking.evaluation.tracker_multi import Tracker  # multi-object tracker
from ltr.models.bbreg.atom_iou_net import AtomIoUNet     # the model (runs forward on CPU)
```

(Demonstrated on saion's login node: `cuda available: False`, no Qt, pure PrRoIPool.)

## Local runner

The repo also has a Windows-friendly runner that applies the same patching in Python,
installs small runtime stubs for legacy optional packages (`jpeg4py`, `visdom`,
`similaritymeasures`), handles torch>=2.6 legacy checkpoint loading, prepares UDMT's
project folder contract, and calls ST-Net directly:

```bash
python -m UDMT.stnet.runner patch-clone --udmt-root Z:/StephensU/medaka/udmt/UDMT
python -m UDMT.stnet.runner verify --udmt-root Z:/StephensU/medaka/udmt/UDMT --run-iou-forward --device cpu
python -m UDMT.stnet.runner run-video \
    --udmt-root Z:/StephensU/medaka/udmt/UDMT \
    --video Z:/StephensU/medaka/medaka_202512/<experiment>/top_cam/cam0_top.mp4 \
    --project runs/udmt_local/<stem> \
    --stem <stem> \
    --start-positions Z:/StephensU/medaka/udmt/bundle/init/top_cam_start_pos.txt \
    --init-mask Z:/StephensU/medaka/udmt/bundle/init/top_cam_init_mask.png \
    --repeat-init-mask \
    --start-frame 176300 --frame-limit 5 --skip-tune --quick
```

For a real run, pass a per-frame `--mask-dir` instead of `--repeat-init-mask`; repeating the
init mask is only a plumbing smoke test.

## Status & remaining work

**Done + validated:** the keystone CUDA→PyTorch op (1e-6 / 1e-8 parity), and the GUI/relative-import
decoupling so the full ST-Net/TrDiMP stack imports standalone on CPU and the IoU-Net runs forward
there.

**Remaining for a fully repo-native standalone tracker** (mechanical, not yet done):
- **Vendor** the decoupled `ltr/` + `pytracking/` into the repo (currently the scripts patch a clone
  in place) — most of the 43k LOC is unused PyTracking baggage, so a trimmed inference-only subset
  is preferable.
- **Guard** the `from PySide6 import QtGui` visualizer imports behind `if vis_flag:` rather than
  stubbing them.
- Wire `UDMT_BASE_DIR` / checkpoint paths and run the *full* tracker (load `trdimp_net_ep.pth.tar`,
  track a clip) standalone end-to-end — so far only import + model-forward are proven off-clone.

> **Note on this machine:** local torch is currently broken (`c10.dll` init failure — a Blackwell/
> CUDA env issue), so the tests run in CI / on saion, not locally. Fixing local torch is what makes
> the portable stack actually runnable on this box — the whole point of the refactor.

## Why this matters here

For medaka, native UDMT/DiMP underperforms the simple motion tracker (~0.1 vs 0.97 coverage — see
`project-udmt-individual-tracking`), so this refactor is **enabling/portability work, not a
recommendation to use ST-Net for medaka**. It removes the Linux/GPU/Qt lock-in so the tracker can be
run, inspected, and compared anywhere (incl. this Blackwell box) without the saion clone.
